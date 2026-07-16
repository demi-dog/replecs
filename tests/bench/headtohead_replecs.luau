--!native
--!optimize 2

-- head-to-head scenario bench (replecs side). The duplecs twin lives at
-- duplecs/tests/benchmarks/bench_headtohead_duplecs.luau; both drive the same
-- scenario shapes and print rows in the same format so the numbers compare
-- directly.
--
-- Scenario: N entities, each carrying Position (number, hot), Health (number,
-- warm), and a static tag. P clients. No serdes on either side, so both
-- libraries carry values out-of-band as Lua values and buffers carry
-- ids/structure only.

local replecs = require "@replecs"
local jecs = require "@pkg/jecs"

local N = 1000
local REPS = 200
local WARM = 20

local guard = 0

local function make_components(world: jecs.World)
	local position = world:component()
	world:set(position, jecs.Name, "position")
	world:add(position, replecs.shared)
	local health = world:component()
	world:set(health, jecs.Name, "health")
	world:add(health, replecs.shared)
	local static_tag = world:entity()
	world:set(static_tag, jecs.Name, "static_tag")
	world:add(static_tag, replecs.shared)
	return position, health, static_tag
end

local function setup(P: number, filter_groups: number?)
	local world = jecs.world()
	local server = replecs.create_server(world)
	server:init()

	local position, health, static_tag = make_components(world)

	local players = table.create(P) :: { any }
	for i = 1, P do
		players[i] = { id = i }
		server.masking:register_client(players[i])
		server.masking:activate_client(players[i])
	end

	-- optional entity-level visibility filters: filter_groups distinct
	-- whitelists of 10 clients each, assigned round-robin
	local groups = nil
	if filter_groups then
		groups = {}
		for g = 1, filter_groups do
			local filter = {}
			for k = 0, 9 do
				filter[players[(g - 1 + k * 3) % P + 1]] = true
			end
			groups[g] = filter
		end
	end

	local entities = table.create(N)
	for i = 1, N do
		local entity = world:entity()
		world:set(entity, position, i * 1.5)
		world:set(entity, health, 100)
		world:add(entity, static_tag)
		server:set_reliable(entity, position)
		server:set_reliable(entity, health)
		server:set_reliable(entity, static_tag)
		if groups then
			server:set_networked(entity, table.clone(groups[(i - 1) % filter_groups :: number + 1]))
		else
			server:set_networked(entity)
		end
		entities[i] = entity
	end

	-- flush the initial additions out of the delta path
	for _player, _buf, _variants in server:collect_updates() do
		guard += 1
	end

	return world, server, position, health, players, entities
end

local function drain(server): (number, number, number)
	-- drains the iterator; returns (packets, bytes, variants) where bytes and
	-- variants are those of the first player seen (one representative client)
	local n = 0
	local bytes = 0
	local variants_n = 0
	for _player, buf, variants in server:collect_updates() do
		n += 1
		if n == 1 then
			bytes = buffer.len(buf)
			for _, inner in variants do
				variants_n += #inner
			end
		end
	end
	return n, bytes, variants_n
end

local function bench_tick(label: string, P: number, changed: number, filter_groups: number?)
	local world, server, position, _health, _players, entities = setup(P, filter_groups)

	local tick = 0
	local function dirty()
		tick += 1
		for i = 1, changed do
			world:set(entities[i], position, i * 1.5 + tick)
		end
	end

	for _ = 1, WARM do
		dirty()
		guard += (drain(server))
	end

	local elapsed = 0
	local bytes, variants = 0, 0
	for _ = 1, REPS do
		dirty()
		local t = os.clock()
		local n, b, v = drain(server)
		elapsed += os.clock() - t
		guard += n
		bytes, variants = b, v
	end

	print(
		string.format(
			"%-34s P=%3d changed=%4d | %8.4f ms/tick | %6d B + %4d values /client",
			label,
			P,
			changed,
			1000 * elapsed / REPS,
			bytes,
			variants
		)
	)
end

local function bench_tick_with_hooks(label: string, P: number, changed: number)
	-- times the world:set mutations too, so per-change hook overhead shows up
	local world, server, position, _health, _players, entities = setup(P, nil)

	local tick = 0
	for _ = 1, WARM do
		tick += 1
		for i = 1, changed do
			world:set(entities[i], position, i * 1.5 + tick)
		end
		guard += (drain(server))
	end

	local elapsed = 0
	for _ = 1, REPS do
		tick += 1
		local t = os.clock()
		for i = 1, changed do
			world:set(entities[i], position, i * 1.5 + tick)
		end
		local n = drain(server)
		elapsed += os.clock() - t
		guard += n
	end
	print(
		string.format(
			"%-34s P=%3d changed=%4d | %8.4f ms/tick (incl. world:set)",
			label,
			P,
			changed,
			1000 * elapsed / REPS
		)
	)
end

local function bench_join(label: string, P: number)
	local _world, server, _position, _health, players, _entities = setup(P, nil)

	local elapsed = 0
	local bytes, variants = 0, 0
	local joins = 50
	for _ = 1, joins do
		local t = os.clock()
		local buf, vars = server:get_full(players[1])
		elapsed += os.clock() - t
		bytes = buffer.len(buf)
		variants = 0
		for _, inner in vars do
			variants += #inner
		end
	end
	print(
		string.format(
			"%-34s P=%3d              | %8.4f ms/join | %6d B + %4d values",
			label,
			P,
			1000 * elapsed / joins,
			bytes,
			variants
		)
	)
end

local function bench_apply(label: string, P: number, changed: number)
	local world, server, position, _health, players, entities = setup(P, nil)

	local c_world = jecs.world()
	local client = replecs.create_client(c_world)
	client:init()
	make_components(c_world)

	local full_buf, full_vars = server:get_full(players[1])
	client:apply_full(full_buf, full_vars)

	local tick = 0
	local function dirty()
		tick += 1
		for i = 1, changed do
			world:set(entities[i], position, i * 1.5 + tick)
		end
	end

	local elapsed = 0
	for rep = 1, WARM + REPS do
		dirty()
		for player, buf, variants in server:collect_updates() do
			if player == players[1] then
				if rep <= WARM then
					client:apply_updates(buf, variants)
				else
					local t = os.clock()
					client:apply_updates(buf, variants)
					elapsed += os.clock() - t
				end
			end
		end
	end
	print(string.format("%-34s P=%3d changed=%4d | %8.4f ms/apply (client)", label, P, changed, 1000 * elapsed / REPS))
end

local function bench_tick_serdes(label: string, P: number, changed: number)
	-- fixed-bytespan serdes on the hot component: values pack into the packet
	-- buffer via replecs's buffer-per-value serialize path
	local world, server, position, _health, _players, entities = setup(P, nil)
	server:set_serdes(position, {
		bytespan = 8,
		serialize = function(value: number): buffer
			local b = buffer.create(8)
			buffer.writef64(b, 0, value)
			return b
		end,
		deserialize = function(b: buffer): number
			return buffer.readf64(b, 0)
		end,
	})

	local tick = 0
	local function dirty()
		tick += 1
		for i = 1, changed do
			world:set(entities[i], position, i * 1.5 + tick)
		end
	end

	for _ = 1, WARM do
		dirty()
		guard += (drain(server))
	end

	local elapsed = 0
	local bytes, variants = 0, 0
	for _ = 1, REPS do
		dirty()
		local t = os.clock()
		local n, b, v = drain(server)
		elapsed += os.clock() - t
		guard += n
		bytes, variants = b, v
	end

	print(
		string.format(
			"%-34s P=%3d changed=%4d | %8.4f ms/tick | %6d B + %4d values /client",
			label,
			P,
			changed,
			1000 * elapsed / REPS,
			bytes,
			variants
		)
	)
end

local function bench_unreliable(label: string, P: number)
	-- replecs-only: position marked unreliable instead of reliable; the channel
	-- resends every marked value each collect, no change tracking involved
	local world = jecs.world()
	local server = replecs.create_server(world)
	server:init()

	local position, health, static_tag = make_components(world)
	-- replecs recommends serdes for every unreliable component so packet
	-- splitting can size values exactly (duplecs's channel requires it)
	server:set_serdes(position, {
		bytespan = 8,
		serialize = function(value: number): buffer
			local b = buffer.create(8)
			buffer.writef64(b, 0, value)
			return b
		end,
		deserialize = function(b: buffer): number
			return buffer.readf64(b, 0)
		end,
	})

	local players = table.create(P) :: { any }
	for i = 1, P do
		players[i] = { id = i }
		server.masking:register_client(players[i])
		server.masking:activate_client(players[i])
	end

	local entities = table.create(N)
	for i = 1, N do
		local entity = world:entity()
		world:set(entity, position, i * 1.5)
		world:set(entity, health, 100)
		world:add(entity, static_tag)
		server:set_unreliable(entity, position)
		server:set_reliable(entity, health)
		server:set_reliable(entity, static_tag)
		server:set_networked(entity)
		entities[i] = entity
	end
	for _p, _b, _v in server:collect_updates() do
		guard += 1
	end

	local tick = 0
	local function dirty()
		tick += 1
		for i = 1, N do
			world:set(entities[i], position, i * 1.5 + tick)
		end
	end

	local function drain_unreliable(): (number, number, number)
		local n = 0
		local bytes = 0
		local variants_n = 0
		for player, buf, variants in server:collect_unreliable() do
			n += 1
			if player == players[1] then
				bytes += buffer.len(buf)
				for _, inner in variants do
					variants_n += #inner
				end
			end
		end
		return n, bytes, variants_n
	end

	for _ = 1, WARM do
		dirty()
		guard += (drain_unreliable())
	end

	local elapsed = 0
	local bytes, variants = 0, 0
	for _ = 1, REPS do
		dirty()
		local t = os.clock()
		local n, b, v = drain_unreliable()
		elapsed += os.clock() - t
		guard += n
		bytes, variants = b, v
	end

	print(
		string.format(
			"%-34s P=%3d changed=%4d | %8.4f ms/tick | %6d B + %4d values /client",
			label,
			P,
			N,
			1000 * elapsed / REPS,
			bytes,
			variants
		)
	)
end

print(string.format("replecs head-to-head  N=%d entities, reps=%d", N, REPS))
bench_tick("tick: all positions change", 30, N, nil)
bench_tick("tick: 10% positions change", 30, N / 10, nil)
bench_tick("tick: all change, 100 clients", 100, N, nil)
bench_tick("tick: all change, filtered vis", 30, N, 10)
bench_tick_with_hooks("tick incl. mutation hooks", 30, N)
bench_join("join: full snapshot", 30)
bench_apply("client apply: all change", 30, N)
bench_tick_serdes("tick: all change, serdes position", 30, N)
bench_unreliable("unreliable tick: all positions", 30)
print("guard:", guard)

return nil
