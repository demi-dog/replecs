--!native
--!optimize 2

-- replecs end-to-end overhead benchmark — the replecs port of duplecs's
-- tests/benchmarks/bench_overhead.luau, with the same workload shapes,
-- constants, and row layout so the two libraries' outputs can be compared
-- side by side. Every server row runs one mutation workload twice: on a plain
-- jecs world with replecs absent entirely (baseline; the mutations alone are
-- timed) and on a replecs server world (the mutations plus the same frame's
-- collect_updates — or collect_unreliable — timed together), printing both
-- times and the ratio. The ratio answers "how much more does this operation
-- cost end-to-end once it replicates than it does on a world that isn't
-- networked at all", amortized per op at the row's stated ops/frame.
--
-- Mapping notes (where replecs's model differs from duplecs's):
--   * replecs tracks replication per entity per component
--     (server:set_reliable(entity, component)), where duplecs marks the
--     component once globally (net.Networked) and entities with a Replicated
--     tag. For persistent cells the tracking calls happen in setup (untimed);
--     for the spawn+despawn rows each spawn's set_networked + set_reliable
--     calls are part of the timed op, since they are the per-op API cost of
--     replicating a fresh entity.
--   * collect_updates returns an iterator that builds each member's combined
--     packet lazily, so the timed region includes draining it (duplecs's
--     generate_packets returns finished packets; its drain is a bare count
--     and stays untimed).
--   * every server row also prints a mutate/collect split of the same timed
--     frames: `mutate` is the workload's mutations with replecs's hooks live
--     (mutate minus plain ~ the per-op capture/staging cost) and `collect` is
--     the packet build plus drain — so a row says not just how much overhead
--     replication adds but which phase to optimize. The joiner row splits the
--     same way into activate (mask migration) and get_full (snapshot build).
--   * replecs has no equivalent of duplecs's NetworkedOnce tracking mode, so
--     that row is absent here.
--   * collect_unreliable re-sends every active unreliable value each call
--     rather than staging changes; the workloads here churn every cell every
--     frame, so the two channels do equivalent per-frame work.
--   * duplecs's per-cell whitelists (pair(net.Private, component)) map to
--     replecs include filters (set_reliable(entity, component, filter)).
--
-- Sections (same order as duplecs's bench):
--   canonical      — the headline rows at P=32 (broadcast visibility,
--     serdes-less values): value sets under each tracking mode,
--     component/tag toggles, pair sets and retargets, spawn+despawn. Each row
--     re-measures its own plain twin, so the plain column doubles as a
--     per-run noise gauge.
--   clients        — the value-set and spawn+despawn rows swept across client
--     counts, against one shared plain baseline per sweep.
--   fragmentation  — value churn under per-cell include filters drawn from an
--     F-mask pool (F distinct visibility groups — the fragmentation axis),
--     then a fan-out sweep (mask size j at pinned F), same shared-baseline
--     arrangement.
--   floor          — a settled world's collect_updates alone: the per-frame
--     fixed cost with nothing staged, per client count.
--   joiner         — one client's full packet built from a populated world
--     (absolute ms/join; joining has no plain-jecs analog, so no ratio). The
--     timed region is register_client + activate_client + get_full — in
--     replecs activation is what moves the world's entities into the new
--     member's visibility masks, so it belongs to the join.
--   client         — apply_updates/apply_unreliable/apply_full timed against
--     applying the same changes to a plain world directly with world:set and
--     friends; the server side of each link runs untimed.
--
-- No serdes except on the unreliable rows (mirroring the duplecs bench, whose
-- unreliable channel requires one), so reliable values ride the variants side
-- array. User ids are padded to the same fixed floors as the duplecs bench so
-- neither library's internal id count can swing cross-bench comparisons
-- through hash-order artifacts.

local jecs = require "../../jecs"
local replecs = require "../../src"
local pair = jecs.pair

local CELLS = 1024 -- value-churn cells (one set each per frame)
local TOGGLES = 512 -- toggled entities (one remove+add cycle each per frame)
local PAIR_CELLS = 512 -- pair-holding subjects (one pair set each per frame)
local SPAWN_RING = 256 -- ring of short-lived entities
local SPAWNS = 128 -- delete+respawn cycles per frame
local FULL_BASE = 1024 -- joiner-world entities (shape mirrors duplecs's full)
local FULL_PAIRS = 256 -- joiner-world pair cells
local CHURN_LIVE = 1024 -- client-section churn ring size
local CHURN_PER_FRAME = 128 -- client-section delete+respawn cycles per frame

local DEFAULT_P = 32
local CLIENT_COUNTS = { 8, 32, 128 }
local FRAG_P = 64 -- the fragmentation and fan-out sweeps' client count
local MASK_COUNTS = { 1, 4, 16, 64, 256 } -- F sweep at j = FRAG_P / 4
local VIEWER_COUNTS = { 2, 16, 48 } -- j sweep at F = 16

local REPS = 200
local WARM = 20
local FULL_REPS = 100

-- fixed id floors (see the header): user components and entities always start
-- here no matter how many ids replecs claims internally
local COMPONENT_FLOOR = 32
local ENTITY_FLOOR = 4096

-- the one serdes shape in the file, used by the unreliable rows to mirror the
-- duplecs bench's f64 hook
local f64_serdes = {
	bytespan = 8,
	serialize = function(value: number)
		local encoded = buffer.create(8)
		buffer.writef64(encoded, 0, value)
		return encoded
	end,
	deserialize = function(encoded: buffer)
		return buffer.readf64(encoded, 0)
	end,
}

-- ---- distinct mask generation: `count` unique random j-subsets of P member
-- indices (0-based), deduped by their numeric bitmask. Identical to the
-- duplecs bench (same seed, same algorithm); C(P, j) stays well above every
-- swept count, so the retry loop is effectively free.

local function generate_masks(count, j, P)
	math.randomseed(1337)
	local masks = table.create(count)
	local seen = {}
	while #masks < count do
		local deck = table.create(P)
		for i = 1, P do
			deck[i] = i - 1
		end
		local key = 0
		local indices = table.create(j)
		for k = 1, j do
			local pick = math.random(k, P)
			deck[k], deck[pick] = deck[pick], deck[k]
			indices[k] = deck[k]
			key += 2 ^ deck[k]
		end
		if seen[key] == nil then
			seen[key] = true
			table.insert(masks, indices)
		end
	end
	return masks
end

-- ---- guard: count every drained packet so the collect calls can't be
-- optimized away. belt-and-braces.

local guard = 0

local function drain(iterator)
	local n = 0
	for _member, _buf, _variants in iterator do
		n += 1
	end
	return n
end

-- ---- shared registration: every replicated definition must be a shared id
-- (replecs.shared + jecs.Name) so the wire format can encode it; the plain
-- twin never calls this. Names only need to be unique per world.

local share_counter = 0

local function share(world, id)
	share_counter += 1
	world:add(id, replecs.shared)
	world:set(id, jecs.Name, "bench_shared_" .. share_counter)
end

-- ---- world construction: P = nil builds the plain twin (no replecs contact
-- at all), a number builds a replecs server world with P registered and
-- activated members. Both sides pad user ids to the fixed floors.

local function make_world(P)
	local world = jecs.world()
	local server = nil
	if P ~= nil then
		server = replecs.create_server(world)
		server:init()
	end

	while (world:component() :: any) < COMPONENT_FLOOR do
	end
	while (world:entity() :: any) < ENTITY_FLOOR do
	end

	local members = nil
	if server ~= nil then
		members = table.create(P) :: { any }
		for i = 1, P do
			members[i] = {}
			server.masking:register_client(members[i])
			server.masking:activate_client(members[i])
		end
		guard += drain(server:collect_updates())
	end

	return world, server, members
end

-- ---- timing shell: WARM then REPS frames of mutate plus the collect call
-- when one is given. The drain runs inside the timed region because replecs
-- materializes each member's packet in the iterator (see the header). The two
-- phases of each frame are clocked separately so every row can report where
-- its overhead lives: `mutate` is the workload's mutations with replecs's
-- hooks live (so mutate minus the plain column is the per-op capture/staging
-- cost), `collect` is the packet build plus drain. Returns total, mutate, and
-- collect nanoseconds per op.

local function measure(mutate, collect, ops)
	for _ = 1, WARM do
		mutate()
		if collect ~= nil then
			guard += drain(collect())
		end
	end

	local mutate_elapsed = 0
	local collect_elapsed = 0
	for _ = 1, REPS do
		local t = os.clock()
		mutate()
		local mid = os.clock()
		mutate_elapsed += mid - t
		if collect ~= nil then
			local n = drain(collect())
			collect_elapsed += os.clock() - mid
			guard += n
		end
	end

	local scale = 1e9 / (REPS * ops)
	return (mutate_elapsed + collect_elapsed) * scale, mutate_elapsed * scale, collect_elapsed * scale
end

local function row(label, P, plain_ns, replecs_ns, mutate_ns, collect_ns)
	print(
		string.format(
			"%-44s P=%3d | plain %8.1f ns/op | replecs %8.1f ns/op | x%5.2f | mutate %8.1f + collect %8.1f",
			label,
			P,
			plain_ns,
			replecs_ns,
			replecs_ns / plain_ns,
			mutate_ns,
			collect_ns
		)
	)
end

-- ---- scenario setups: each builds its workload into `world` (skipping every
-- replecs touch when server is nil — the plain twin) and returns the
-- per-frame mutate closure. Workload shapes are the duplecs bench's.

-- value churn: CELLS entities, one data component each, every cell set fresh
-- per frame. mode picks the tracking channel; the unreliable mode registers
-- the same f64 serdes the duplecs bench uses there.
local function setup_value_churn(world, server, mode)
	local component = world:component()
	if server ~= nil then
		share(world, component)
		if mode == "unreliable" then
			server:set_serdes(component, table.clone(f64_serdes))
		end
	end

	local entities = table.create(CELLS)
	for i = 1, CELLS do
		local entity = world:entity()
		if server ~= nil then
			server:set_networked(entity)
			if mode == "unreliable" then
				server:set_unreliable(entity, component)
			else
				server:set_reliable(entity, component)
			end
		end
		world:set(entity, component, i)
		entities[i] = entity
	end

	local tick = 0
	local function churn()
		tick += 1
		for i = 1, CELLS do
			world:set(entities[i], component, i + tick)
		end
	end

	return churn, entities, component
end

-- pair value churn: PAIR_CELLS subjects each holding one data pair on a
-- shared target, every pair set fresh per frame. Pairs with entity targets
-- replicate through relation tracking (set_relation) in replecs; set_pair is
-- for shared-id targets only.
local function setup_pair_churn(world, server)
	local relation = world:component()
	local target = world:entity()
	if server ~= nil then
		share(world, relation)
		server:set_networked(target)
	end

	local subjects = table.create(PAIR_CELLS)
	for i = 1, PAIR_CELLS do
		local subject = world:entity()
		if server ~= nil then
			server:set_networked(subject)
			server:set_relation(subject, relation)
		end
		world:set(subject, pair(relation, target), i)
		subjects[i] = subject
	end

	local tick = 0
	local function churn()
		tick += 1
		for i = 1, PAIR_CELLS do
			world:set(subjects[i], pair(relation, target), i + tick)
		end
	end

	return churn
end

-- component toggles: TOGGLES entities; each op removes the component and sets
-- (or, for tags, adds) it back — two archetype moves per op
local function setup_toggle(world, server, tag_definition)
	local component
	if tag_definition then
		component = world:entity()
	else
		component = world:component()
	end
	if server ~= nil then
		share(world, component)
	end

	local entities = table.create(TOGGLES)
	for i = 1, TOGGLES do
		local entity = world:entity()
		if server ~= nil then
			server:set_networked(entity)
			server:set_reliable(entity, component)
		end
		if tag_definition then
			world:add(entity, component)
		else
			world:set(entity, component, i)
		end
		entities[i] = entity
	end

	local tick = 0
	local function toggle()
		tick += 1
		if tag_definition then
			for i = 1, TOGGLES do
				world:remove(entities[i], component)
				world:add(entities[i], component)
			end
		else
			for i = 1, TOGGLES do
				world:remove(entities[i], component)
				world:set(entities[i], component, i + tick)
			end
		end
	end

	return toggle
end

-- pair retargets: TOGGLES subjects alternating between two targets; each op
-- removes pair(relation, old) and sets pair(relation, new). replecs tracks
-- the whole relation (set_relation), which is exactly its retarget-following
-- mode.
local function setup_retarget(world, server)
	local relation = world:component()
	local targets = { world:entity(), world:entity() }
	if server ~= nil then
		share(world, relation)
		server:set_networked(targets[1])
		server:set_networked(targets[2])
	end

	local subjects = table.create(TOGGLES)
	local assigned = table.create(TOGGLES)
	for i = 1, TOGGLES do
		local subject = world:entity()
		local target_index = (i - 1) % 2 + 1
		if server ~= nil then
			server:set_networked(subject)
			server:set_relation(subject, relation)
		end
		world:set(subject, pair(relation, targets[target_index]), i)
		subjects[i] = subject
		assigned[i] = target_index
	end

	local tick = 0
	local function rotate()
		tick += 1
		for i = 1, TOGGLES do
			local old_index = assigned[i]
			local new_index = old_index % 2 + 1
			world:remove(subjects[i], pair(relation, targets[old_index]))
			world:set(subjects[i], pair(relation, targets[new_index]), i + tick)
			assigned[i] = new_index
		end
	end

	return rotate
end

-- spawn+despawn: a ring of short-lived entities each holding two data
-- components; every op deletes the oldest slot and respawns it (recycled
-- ids). The replecs side's per-spawn tracking calls (set_networked plus one
-- set_reliable per cell) are part of the op — see the header.
local function setup_spawn_despawn(world, server)
	local component_a = world:component()
	local component_b = world:component()
	if server ~= nil then
		share(world, component_a)
		share(world, component_b)
	end

	local tick = 0
	local ring = table.create(SPAWN_RING)
	local function spawn(slot)
		tick += 1
		local entity = world:entity()
		if server ~= nil then
			server:set_networked(entity)
			server:set_reliable(entity, component_a)
			server:set_reliable(entity, component_b)
		end
		world:set(entity, component_a, tick)
		world:set(entity, component_b, tick * 2)
		ring[slot] = entity
	end
	for slot = 1, SPAWN_RING do
		spawn(slot)
	end

	local slot_cursor = 0
	local function churn()
		for _ = 1, SPAWNS do
			slot_cursor = slot_cursor % SPAWN_RING + 1
			world:delete(ring[slot_cursor])
			spawn(slot_cursor)
		end
	end

	return churn
end

-- ---- server rows ------------------------------------------------------------

-- measures one setup's plain twin, ready for the replecs side of a row (or a
-- sweep of them) to compare against
local function measure_plain(setup, ops, ...)
	local world = make_world(nil)
	local mutate = setup(world, nil, ...)
	return measure(mutate, nil, ops)
end

-- one self-contained row: the plain twin, then the replecs side with the
-- setup's additions flushed so the timed frames diff a settled world
local function run_server_row(label, P, setup, ops, ...)
	local plain_ns = measure_plain(setup, ops, ...)

	local world, server = make_world(P)
	local mutate = setup(world, server, ...)
	guard += drain(server:collect_updates())

	local collect = function()
		return server:collect_updates()
	end
	row(label, P, plain_ns, measure(mutate, collect, ops))
end

-- the value-set row under each tracking mode, sharing one plain twin (the
-- plain workload is identical across modes). The unreliable mode times
-- collect_unreliable instead — its values never ride the reliable deltas,
-- and the settle call has already synced the additions.
local function run_value_row(label, P, mode, plain_ns)
	local world, server = make_world(P)
	local mutate = setup_value_churn(world, server, mode)
	guard += drain(server:collect_updates())

	local collect
	if mode == "unreliable" then
		collect = function()
			return server:collect_unreliable()
		end
	else
		collect = function()
			return server:collect_updates()
		end
	end

	row(label, P, plain_ns, measure(mutate, collect, CELLS))
end

-- fragmentation: the reliable value-churn row with a per-cell component
-- include filter drawn round-robin from a pool of F distinct j-sized masks,
-- so F sweeps visibility-group count and j sweeps fan-out
local function run_fragmented_row(F, j, plain_ns)
	local world, server, members = make_world(FRAG_P)
	local mutate, entities, component = setup_value_churn(world, server, "reliable")

	local masks = generate_masks(F, j, FRAG_P)
	for i = 1, CELLS do
		local indices = masks[(i - 1) % F + 1]
		local filter = {}
		for _, index in indices do
			filter[members[index + 1]] = true
		end
		server:set_reliable(entities[i], component, filter)
	end
	guard += drain(server:collect_updates())

	local collect = function()
		return server:collect_updates()
	end
	local label = string.format("set: reliable, F=%3d masks of j=%2d", F, j)
	row(label, FRAG_P, plain_ns, measure(mutate, collect, CELLS))
end

-- floor: a fully settled world's collect_updates with nothing staged — the
-- per-frame fixed cost that every ratio above amortizes over its ops
local function run_floor_row(P)
	local world, server = make_world(P)
	local mutate = setup_value_churn(world, server, "reliable")
	mutate()
	guard += drain(server:collect_updates())

	local collect = function()
		return server:collect_updates()
	end
	local noop = function() end
	local ns = measure(noop, collect, 1)
	print(string.format("settled collect_updates                      P=%3d | %8.2f us/frame", P, ns / 1e3))
end

-- joiner: one client's full packet from a populated world — register_client
-- + activate_client plus the get_full that builds the snapshot, with the
-- departure untimed so the world holds one extra member at a time
local function run_joiner_row()
	local world, server = make_world(0)

	local components = table.create(4)
	for c = 1, 4 do
		components[c] = world:component()
		share(world, components[c])
	end
	local tag = world:entity()
	share(world, tag)
	local relation = world:component()
	share(world, relation)

	local entities = table.create(FULL_BASE)
	for i = 1, FULL_BASE do
		local entity = world:entity()
		server:set_networked(entity)
		local component = components[(i - 1) % 4 + 1]
		server:set_reliable(entity, component)
		world:set(entity, component, i)
		if i % 4 == 0 then
			server:set_reliable(entity, tag)
			world:add(entity, tag)
		end
		entities[i] = entity
	end
	for i = 1, FULL_PAIRS do
		local subject = entities[FULL_PAIRS + i]
		server:set_relation(subject, relation)
		world:set(subject, pair(relation, entities[i]), i * 3)
	end
	guard += drain(server:collect_updates())

	local activate_elapsed = 0
	local full_elapsed = 0
	local bytes = 0
	for _ = 1, FULL_REPS do
		local viewer = {}
		local t = os.clock()
		server.masking:register_client(viewer)
		server.masking:activate_client(viewer)
		local mid = os.clock()
		local buf = server:get_full(viewer)
		full_elapsed += os.clock() - mid
		activate_elapsed += mid - t
		bytes = buffer.len(buf)
		server.masking:unregister_client(viewer)
	end

	local cells = FULL_BASE + FULL_BASE // 4 + FULL_PAIRS
	print(
		string.format(
			"full packet: %4d entities, %4d cells        | %8.4f ms/join | %6d B | activate %.4f + get_full %.4f",
			FULL_BASE,
			cells,
			1000 * (activate_elapsed + full_elapsed) / FULL_REPS,
			bytes,
			1000 * activate_elapsed / FULL_REPS,
			1000 * full_elapsed / FULL_REPS
		)
	)
end

-- ---- client section ----------------------------------------------------------

-- one server/client pairing plus the plain twin world the direct-application
-- baseline mutates; the viewer stand-in is only ever a table key. Shared
-- components resolve across the two worlds by name (replecs.shared +
-- jecs.Name), so map_component/map_tag name each definition identically on
-- both sides before the init calls.
local function make_link()
	local server_world = jecs.world()
	local server = replecs.create_server(server_world)
	while (server_world:component() :: any) < COMPONENT_FLOOR do
	end
	while (server_world:entity() :: any) < ENTITY_FLOOR do
	end

	local client_world = jecs.world()
	local client = replecs.create_client(client_world)
	while (client_world:component() :: any) < COMPONENT_FLOOR do
	end
	while (client_world:entity() :: any) < ENTITY_FLOOR do
	end

	local plain_world = make_world(nil)

	return {
		server_world = server_world,
		server = server,
		client_world = client_world,
		client = client,
		plain_world = plain_world,
		viewer = {},
		mapped = 0,
	}
end

-- a data component on both worlds under one shared name, so its sections
-- resolve locally, plus the plain twin's own component
local function map_component(link)
	link.mapped += 1
	local name = "bench_component_" .. link.mapped
	local server_component = link.server_world:component()
	link.server_world:add(server_component, replecs.shared)
	link.server_world:set(server_component, jecs.Name, name)
	local client_component = link.client_world:component()
	link.client_world:add(client_component, replecs.shared)
	link.client_world:set(client_component, jecs.Name, name)
	local plain_component = link.plain_world:component()
	return server_component, client_component, plain_component
end

-- same for a valueless tag
local function map_tag(link)
	link.mapped += 1
	local name = "bench_tag_" .. link.mapped
	local server_tag = link.server_world:entity()
	link.server_world:add(server_tag, replecs.shared)
	link.server_world:set(server_tag, jecs.Name, name)
	local client_tag = link.client_world:entity()
	link.client_world:add(client_tag, replecs.shared)
	link.client_world:set(client_tag, jecs.Name, name)
	local plain_tag = link.plain_world:entity()
	return server_tag, client_tag, plain_tag
end

-- inits both halves (after the components exist) and registers the viewer
local function finish_link(link)
	link.server:init()
	link.client:init()
	link.server.masking:register_client(link.viewer)
	link.server.masking:activate_client(link.viewer)
end

-- settles the setup's additions, then seeds the client's entity map from the
-- viewer's full packet (untimed)
local function sync_initial(link)
	guard += drain(link.server:collect_updates())
	local buf, variants = link.server:get_full(link.viewer)
	link.client:apply_full(buf, variants)
end

-- pulls the viewer's one combined update packet off the iterator
local function viewer_packet(link)
	local buf, variants
	for _member, b, v in link.server:collect_updates() do
		buf, variants = b, v
	end
	return buf, variants
end

-- times REPS frames of the apply side (dirty mutates the server world
-- untimed, the viewer's packet applies timed) against REPS frames of the
-- plain twin's direct application, and prints the row
local function run_client_row(link, label, ops, dirty, apply)
	for rep = 1, WARM do
		apply(rep)
	end
	local plain_elapsed = 0
	for rep = 1, REPS do
		local t = os.clock()
		apply(rep)
		plain_elapsed += os.clock() - t
	end

	local client = link.client
	for rep = 1, WARM do
		dirty(rep)
		local buf, variants = viewer_packet(link)
		client:apply_updates(buf, variants)
	end
	local elapsed = 0
	for rep = 1, REPS do
		dirty(rep)
		local buf, variants = viewer_packet(link)
		local t = os.clock()
		client:apply_updates(buf, variants)
		elapsed += os.clock() - t
	end

	local plain_ns = 1e9 * plain_elapsed / (REPS * ops)
	local replecs_ns = 1e9 * elapsed / (REPS * ops)
	print(
		string.format(
			"%-32s ops=%5d | direct %8.1f ns/op | apply %8.1f ns/op | x%5.2f",
			label,
			ops,
			plain_ns,
			replecs_ns,
			replecs_ns / plain_ns
		)
	)
end

-- values: CELLS unpaired cells churned per frame on both sides of the wire
-- and on the plain twin
local function run_client_values()
	local link = make_link()
	local server_world = link.server_world
	local plain_world = link.plain_world
	local server_component, _, plain_component = map_component(link)
	finish_link(link)

	local entities = table.create(CELLS)
	local plain_entities = table.create(CELLS)
	for i = 1, CELLS do
		local entity = server_world:entity()
		link.server:set_networked(entity)
		link.server:set_reliable(entity, server_component)
		server_world:set(entity, server_component, i)
		entities[i] = entity
		local plain_entity = plain_world:entity()
		plain_world:set(plain_entity, plain_component, i)
		plain_entities[i] = plain_entity
	end
	sync_initial(link)

	run_client_row(link, "values", CELLS, function(rep)
		for i = 1, CELLS do
			server_world:set(entities[i], server_component, i * 7 + rep)
		end
	end, function(rep)
		for i = 1, CELLS do
			plain_world:set(plain_entities[i], plain_component, i * 7 + rep)
		end
	end)
end

-- adds_removes: TOGGLES entities, half adding and half removing the component
-- each frame on both sides
local function run_client_adds_removes()
	local link = make_link()
	local server_world = link.server_world
	local plain_world = link.plain_world
	local server_component, _, plain_component = map_component(link)
	finish_link(link)

	local entities = table.create(TOGGLES)
	local plain_entities = table.create(TOGGLES)
	for i = 1, TOGGLES do
		local entity = server_world:entity()
		link.server:set_networked(entity)
		link.server:set_reliable(entity, server_component)
		local plain_entity = plain_world:entity()
		if i % 2 == 0 then
			server_world:set(entity, server_component, i)
			plain_world:set(plain_entity, plain_component, i)
		end
		entities[i] = entity
		plain_entities[i] = plain_entity
	end
	sync_initial(link)

	run_client_row(link, "adds_removes", TOGGLES, function(rep)
		for i = 1, TOGGLES do
			if (i + rep) % 2 == 0 then
				server_world:set(entities[i], server_component, i + rep)
			else
				server_world:remove(entities[i], server_component)
			end
		end
	end, function(rep)
		for i = 1, TOGGLES do
			if (i + rep) % 2 == 0 then
				plain_world:set(plain_entities[i], plain_component, i + rep)
			else
				plain_world:remove(plain_entities[i], plain_component)
			end
		end
	end)
end

-- churn: a ring of live entities; each frame deletes the oldest slice and
-- creates a fresh one in its place, each entity holding one data component
-- and one tag on both sides
local function run_client_churn()
	local link = make_link()
	local server_world = link.server_world
	local plain_world = link.plain_world
	local server_component, _, plain_component = map_component(link)
	local server_tag, _, plain_tag = map_tag(link)
	finish_link(link)

	local live = table.create(CHURN_LIVE)
	local plain_live = table.create(CHURN_LIVE)
	local function spawn(slot, value)
		local entity = server_world:entity()
		link.server:set_networked(entity)
		link.server:set_reliable(entity, server_component)
		link.server:set_reliable(entity, server_tag)
		server_world:set(entity, server_component, value)
		server_world:add(entity, server_tag)
		live[slot] = entity
	end
	local function plain_spawn(slot, value)
		local entity = plain_world:entity()
		plain_world:set(entity, plain_component, value)
		plain_world:add(entity, plain_tag)
		plain_live[slot] = entity
	end
	for i = 1, CHURN_LIVE do
		spawn(i, i)
		plain_spawn(i, i)
	end
	sync_initial(link)

	local cursor = 0
	local plain_cursor = 0
	run_client_row(link, "churn", CHURN_PER_FRAME * 2, function(rep)
		for k = 1, CHURN_PER_FRAME do
			local slot = cursor + k
			server_world:delete(live[slot])
			spawn(slot, rep * CHURN_PER_FRAME + k)
		end
		cursor = (cursor + CHURN_PER_FRAME) % CHURN_LIVE
	end, function(rep)
		for k = 1, CHURN_PER_FRAME do
			local slot = plain_cursor + k
			plain_world:delete(plain_live[slot])
			plain_spawn(slot, rep * CHURN_PER_FRAME + k)
		end
		plain_cursor = (plain_cursor + CHURN_PER_FRAME) % CHURN_LIVE
	end)
end

-- pair values: PAIR_CELLS subjects each holding a data pair to its own
-- target, so one relation spans PAIR_CELLS distinct pair ids on both sides
local function run_client_pair_values()
	local link = make_link()
	local server_world = link.server_world
	local plain_world = link.plain_world
	local server_relation, _, plain_relation = map_component(link)
	finish_link(link)

	local subjects = table.create(PAIR_CELLS)
	local server_pairs = table.create(PAIR_CELLS)
	local plain_subjects = table.create(PAIR_CELLS)
	local plain_pairs = table.create(PAIR_CELLS)
	for i = 1, PAIR_CELLS do
		local target = server_world:entity()
		link.server:set_networked(target)
		local subject = server_world:entity()
		link.server:set_networked(subject)
		link.server:set_relation(subject, server_relation)
		server_world:set(subject, pair(server_relation, target), i)
		subjects[i] = subject
		server_pairs[i] = pair(server_relation, target)
		local plain_target = plain_world:entity()
		local plain_subject = plain_world:entity()
		plain_world:set(plain_subject, pair(plain_relation, plain_target), i)
		plain_subjects[i] = plain_subject
		plain_pairs[i] = pair(plain_relation, plain_target)
	end
	sync_initial(link)

	run_client_row(link, "pair values", PAIR_CELLS, function(rep)
		for i = 1, PAIR_CELLS do
			server_world:set(subjects[i], server_pairs[i], i * 3 + rep)
		end
	end, function(rep)
		for i = 1, PAIR_CELLS do
			plain_world:set(plain_subjects[i], plain_pairs[i], i * 3 + rep)
		end
	end)
end

-- unreliable: CELLS always-replicated f64 cells; the timed apply side applies
-- every chunk of one collect_unreliable call, the plain twin applies the same
-- sets directly. Membership arrived through sync_initial's full packet, so
-- chunk values land on existing cells.
local function run_client_unreliable()
	local link = make_link()
	local server_world = link.server_world
	local plain_world = link.plain_world
	local server_component, client_component, plain_component = map_component(link)
	finish_link(link)
	link.server:set_serdes(server_component, table.clone(f64_serdes))
	link.client:set_serdes(client_component, table.clone(f64_serdes))

	local entities = table.create(CELLS)
	local plain_entities = table.create(CELLS)
	for i = 1, CELLS do
		local entity = server_world:entity()
		link.server:set_networked(entity)
		link.server:set_unreliable(entity, server_component)
		server_world:set(entity, server_component, i)
		entities[i] = entity
		local plain_entity = plain_world:entity()
		plain_world:set(plain_entity, plain_component, i)
		plain_entities[i] = plain_entity
	end
	sync_initial(link)

	local client = link.client
	local function dirty(rep)
		for i = 1, CELLS do
			server_world:set(entities[i], server_component, i * 7 + rep)
		end
	end
	local function apply(rep)
		for i = 1, CELLS do
			plain_world:set(plain_entities[i], plain_component, i * 7 + rep)
		end
	end
	local function collect_chunks()
		local chunks = {}
		for _member, buf, variants in link.server:collect_unreliable() do
			table.insert(chunks, { buf, variants })
		end
		return chunks
	end

	for rep = 1, WARM do
		apply(rep)
	end
	local plain_elapsed = 0
	for rep = 1, REPS do
		local t = os.clock()
		apply(rep)
		plain_elapsed += os.clock() - t
	end

	for rep = 1, WARM do
		dirty(rep)
		for _, chunk in collect_chunks() do
			client:apply_unreliable(chunk[1], chunk[2])
		end
	end
	local elapsed = 0
	for rep = 1, REPS do
		dirty(rep)
		local chunks = collect_chunks()
		local t = os.clock()
		for _, chunk in chunks do
			client:apply_unreliable(chunk[1], chunk[2])
		end
		elapsed += os.clock() - t
	end

	local plain_ns = 1e9 * plain_elapsed / (REPS * CELLS)
	local replecs_ns = 1e9 * elapsed / (REPS * CELLS)
	print(
		string.format(
			"%-32s ops=%5d | direct %8.1f ns/op | apply %8.1f ns/op | x%5.2f",
			"unreliable values",
			CELLS,
			plain_ns,
			replecs_ns,
			replecs_ns / plain_ns
		)
	)
end

-- full: a joiner's snapshot applied into a fresh client world each rep,
-- against building the same content into a fresh plain world directly (world
-- construction, component naming, and init untimed on both sides)
local function run_client_full()
	local server_world = jecs.world()
	local server = replecs.create_server(server_world)
	while (server_world:component() :: any) < COMPONENT_FLOOR do
	end
	while (server_world:entity() :: any) < ENTITY_FLOOR do
	end

	local server_components = table.create(4)
	for c = 1, 4 do
		local component = server_world:component()
		server_world:add(component, replecs.shared)
		server_world:set(component, jecs.Name, "bench_full_component_" .. c)
		server_components[c] = component
	end
	local server_tag = server_world:entity()
	server_world:add(server_tag, replecs.shared)
	server_world:set(server_tag, jecs.Name, "bench_full_tag")
	local server_relation = server_world:component()
	server_world:add(server_relation, replecs.shared)
	server_world:set(server_relation, jecs.Name, "bench_full_relation")
	server:init()

	local entities = table.create(FULL_BASE)
	for i = 1, FULL_BASE do
		local entity = server_world:entity()
		server:set_networked(entity)
		local component = server_components[(i - 1) % 4 + 1]
		server:set_reliable(entity, component)
		server_world:set(entity, component, i)
		if i % 4 == 0 then
			server:set_reliable(entity, server_tag)
			server_world:add(entity, server_tag)
		end
		entities[i] = entity
	end
	for i = 1, FULL_PAIRS do
		local subject = entities[FULL_PAIRS + i]
		server:set_relation(subject, server_relation)
		server_world:set(subject, pair(server_relation, entities[i]), i * 3)
	end

	local viewer = {}
	server.masking:register_client(viewer)
	server.masking:activate_client(viewer)
	guard += drain(server:collect_updates())
	local packet, packet_variants = server:get_full(viewer)

	local plain_elapsed = 0
	for _ = 1, FULL_REPS do
		local plain_world = jecs.world()
		while (plain_world:component() :: any) < COMPONENT_FLOOR do
		end
		while (plain_world:entity() :: any) < ENTITY_FLOOR do
		end
		local plain_components = table.create(4)
		for c = 1, 4 do
			plain_components[c] = plain_world:component()
		end
		local plain_tag = plain_world:entity()
		local plain_relation = plain_world:component()

		local t = os.clock()
		local plain_entities = table.create(FULL_BASE)
		for i = 1, FULL_BASE do
			local entity = plain_world:entity()
			plain_world:set(entity, plain_components[(i - 1) % 4 + 1], i)
			if i % 4 == 0 then
				plain_world:add(entity, plain_tag)
			end
			plain_entities[i] = entity
		end
		for i = 1, FULL_PAIRS do
			plain_world:set(plain_entities[FULL_PAIRS + i], pair(plain_relation, plain_entities[i]), i * 3)
		end
		plain_elapsed += os.clock() - t
	end

	local elapsed = 0
	for _ = 1, FULL_REPS do
		local client_world = jecs.world()
		local client = replecs.create_client(client_world)
		while (client_world:component() :: any) < COMPONENT_FLOOR do
		end
		while (client_world:entity() :: any) < ENTITY_FLOOR do
		end
		for c = 1, 4 do
			local component = client_world:component()
			client_world:add(component, replecs.shared)
			client_world:set(component, jecs.Name, "bench_full_component_" .. c)
		end
		local client_tag = client_world:entity()
		client_world:add(client_tag, replecs.shared)
		client_world:set(client_tag, jecs.Name, "bench_full_tag")
		local client_relation = client_world:component()
		client_world:add(client_relation, replecs.shared)
		client_world:set(client_relation, jecs.Name, "bench_full_relation")
		client:init()

		local t = os.clock()
		client:apply_full(packet, packet_variants)
		elapsed += os.clock() - t
	end

	local cells = FULL_BASE + FULL_BASE // 4 + FULL_PAIRS
	local plain_ns = 1e9 * plain_elapsed / (FULL_REPS * cells)
	local replecs_ns = 1e9 * elapsed / (FULL_REPS * cells)
	print(
		string.format(
			"%-32s ops=%5d | direct %8.1f ns/op | apply %8.1f ns/op | x%5.2f  (%.4f ms/join)",
			"full packet",
			cells,
			plain_ns,
			replecs_ns,
			replecs_ns / plain_ns,
			1000 * elapsed / FULL_REPS
		)
	)
end

-- ---- run ----------------------------------------------------------------------

print(
	string.format(
		"replecs end-to-end overhead  cells=%d toggles=%d pairs=%d spawns=%d reps=%d"
			.. "  (plain = same mutations, no replecs; replecs = mutations + collect + drain, timed together;"
			.. " mutate/collect = the same frames' two phases clocked separately)",
		CELLS,
		TOGGLES,
		PAIR_CELLS,
		SPAWNS,
		REPS
	)
)
print ""

print(string.format("canonical rows (P=%d, broadcast visibility, serdes-less values):", DEFAULT_P))
do
	local plain_set_ns = measure_plain(setup_value_churn, CELLS, nil)
	run_value_row("set: reliable", DEFAULT_P, "reliable", plain_set_ns)
	run_value_row("set: unreliable (+collect chunks)", DEFAULT_P, "unreliable", plain_set_ns)
end
run_server_row("set: reliable pair", DEFAULT_P, setup_pair_churn, PAIR_CELLS)
run_server_row("toggle: reliable data component", DEFAULT_P, setup_toggle, TOGGLES, false)
run_server_row("toggle: reliable tag", DEFAULT_P, setup_toggle, TOGGLES, true)
run_server_row("retarget: relation pair", DEFAULT_P, setup_retarget, TOGGLES)
run_server_row("spawn+despawn: 2 tracked cells", DEFAULT_P, setup_spawn_despawn, SPAWNS)
print ""

print "client-count sweep (broadcast visibility):"
do
	local plain_set_ns = measure_plain(setup_value_churn, CELLS, nil)
	for _, P in CLIENT_COUNTS do
		run_value_row("set: reliable", P, "reliable", plain_set_ns)
	end
	local plain_spawn_ns = measure_plain(setup_spawn_despawn, SPAWNS)
	for _, P in CLIENT_COUNTS do
		local world, server = make_world(P)
		local mutate = setup_spawn_despawn(world, server)
		guard += drain(server:collect_updates())
		local collect = function()
			return server:collect_updates()
		end
		row("spawn+despawn: 2 tracked cells", P, plain_spawn_ns, measure(mutate, collect, SPAWNS))
	end
end
print ""

print(
	string.format(
		"visibility fragmentation (P=%d; F distinct include filters of j clients, per-cell round-robin):",
		FRAG_P
	)
)
do
	local plain_set_ns = measure_plain(setup_value_churn, CELLS, nil)
	for _, F in MASK_COUNTS do
		run_fragmented_row(F, FRAG_P // 4, plain_set_ns)
	end
	for _, j in VIEWER_COUNTS do
		run_fragmented_row(16, j, plain_set_ns)
	end
end
print ""

print "per-frame floor (settled world, nothing staged):"
for _, P in CLIENT_COUNTS do
	run_floor_row(P)
end
print ""

print "joiner (server-side full-packet build):"
run_joiner_row()
print ""

print "client (apply vs applying the same changes directly; serdes-less values unless stated):"
run_client_values()
run_client_adds_removes()
run_client_churn()
run_client_pair_values()
run_client_unreliable()
run_client_full()
print ""

print("guard:", guard)

return true
