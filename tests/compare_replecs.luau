--!optimize 2

-- Head-to-head comparison workload, replecs side. Mirrors
-- tests/benchmarks/compare_duplecs.luau in the duplecs repo: same jecs
-- 0.9.0, same entity/component/client shape, same dirty pattern, timed with
-- os.clock. See that file for the scenario descriptions.

local replecs = require("../src")
local jecs = require("../jecs")
local pair = jecs.pair

local NUM_COMPONENTS = 4
local CELLS_PER_COMPONENT = 250
local REPS = 200

local guard = 0

local function run(label, P, j, dirty_fraction, use_serdes)
	local world = jecs.world()
	local server = replecs.create_server(world)
	server:init()

	local players = table.create(P) :: { any }
	for i = 1, P do
		players[i] = { name = "player" .. i }
	end
	for _, player in players do
		server.masking:register_client(player)
		server:mark_player_ready(player)
	end

	local components = table.create(NUM_COMPONENTS)
	for c = 1, NUM_COMPONENTS do
		local component = world:component()
		world:set(component, jecs.Name, "component" .. c)
		world:add(component, replecs.shared)
		if use_serdes then
			world:set(component, replecs.serdes, {
				bytespan = 8,
				serialize = function(value)
					local buf = buffer.create(8)
					buffer.writef64(buf, 0, value)
					return buf
				end,
				deserialize = function(buf)
					return buffer.readf64(buf, 0)
				end,
			})
		end
		components[c] = component
	end

	local cell_entities = table.create(NUM_COMPONENTS * CELLS_PER_COMPONENT)
	local cell_components = table.create(NUM_COMPONENTS * CELLS_PER_COMPONENT)
	local cell_count = 0
	for c = 1, NUM_COMPONENTS do
		local component = components[c]
		for _ = 1, CELLS_PER_COMPONENT do
			cell_count += 1
			local entity = world:entity()
			world:set(entity, component, cell_count * 3)
			world:add(entity, replecs.networked)
			world:add(entity, pair(replecs.reliable, component))
			cell_entities[cell_count] = entity
			cell_components[cell_count] = component
		end
	end

	-- drain the additions backlog so the timed loop starts from steady state
	for _player, _buf, _variants in server:collect_updates() do
	end

	-- per-cell whitelists (j viewers spread across P players)
	if j then
		for cell = 1, cell_count do
			local whitelist = {}
			local stride = math.max(1, math.floor(P / j))
			local offset = (cell * 7) % P
			for k = 0, j - 1 do
				whitelist[players[(offset + k * stride) % P + 1]] = true
			end
			world:set(cell_entities[cell], pair(replecs.reliable, cell_components[cell]), whitelist)
		end
		for _player, _buf, _variants in server:collect_updates() do
		end
	end

	local dirty_count = math.floor(cell_count * dirty_fraction)
	local tick = 0
	local dirty_elapsed = 0
	local packet_elapsed = 0
	local total_bytes = 0
	local total_oob = 0

	for _ = 1, REPS do
		tick += 1
		local t0 = os.clock()
		for cell = 1, dirty_count do
			world:set(cell_entities[cell], cell_components[cell], cell * 3 + tick)
		end
		local t1 = os.clock()
		for player, buf, variants in server:collect_updates() do
			total_bytes += buffer.len(buf)
			guard += 1
			for _, packet_variants in variants do
				total_oob += #packet_variants
			end
		end
		local t2 = os.clock()
		dirty_elapsed += t1 - t0
		packet_elapsed += t2 - t1
	end

	print(string.format(
		"%-28s P=%3d dirty=%4d | dirty %7.3fms  packets %7.3fms  | %8.1f B/client/frame  oob %6.1f/client/frame",
		label,
		P,
		dirty_count,
		1000 * dirty_elapsed / REPS,
		1000 * packet_elapsed / REPS,
		total_bytes / REPS / P,
		total_oob / REPS / P
	))

	-- full snapshot for one late joiner (activation + snapshot, the documented
	-- join flow; duplecs' equivalent is add_client + generate_packets)
	local joiner = { name = "joiner" }
	server.masking:register_client(joiner)
	local t0 = os.clock()
	server:mark_player_ready(joiner)
	local buf, variants = server:get_full(joiner)
	local t1 = os.clock()
	guard += #variants
	print(string.format(
		"%-28s        full snapshot: %6.3fms  %d bytes",
		"",
		1000 * (t1 - t0),
		buffer.len(buf)
	))

	server:destroy()
end

print(string.format("replecs | components=%d cells/comp=%d reps=%d", NUM_COMPONENTS, CELLS_PER_COMPONENT, REPS))
run("broadcast", 8, nil, 1.0, false)
run("broadcast", 32, nil, 1.0, false)
run("broadcast", 64, nil, 1.0, false)
run("broadcast 10% dirty", 32, nil, 0.1, false)
run("filtered j=4", 32, 4, 1.0, false)
run("filtered j=16", 32, 16, 1.0, false)
run("broadcast serdes", 32, nil, 1.0, true)
print("guard:", guard)

return true
