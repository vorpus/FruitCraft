// pybind11 bridge over OpenSnowstorm's mini-openbwapi.
//
// One Brood War game per process (the engine keeps a global Game singleton).
// The engine loads MPQ data files from the process working directory and reads
// ./bwapi-data/bwapi.ini on match start; the Python wrapper manages both.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "openbwapi.h"

namespace py = pybind11;
using namespace OpenBWAPI;

namespace {

Game& game() { return *Broodwar; }

// Only these command types are wired up in mini-openbwapi; anything else
// calls the engine's fatal error handler, so refuse them here instead.
bool command_supported(int type) {
    switch (type) {
    case UnitCommandTypes::Attack_Move:
    case UnitCommandTypes::Attack_Unit:
    case UnitCommandTypes::Move:
    case UnitCommandTypes::Build:
    case UnitCommandTypes::Train:
    case UnitCommandTypes::Right_Click_Unit:
        return true;
    default:
        return false;
    }
}

py::dict unit_to_dict(Unit u) {
    py::dict d;
    d["id"] = u->getID();
    d["type"] = u->getType().getID();
    Position pos = u->getPosition();
    d["x"] = pos.x;
    d["y"] = pos.y;
    d["hp"] = u->getHitPoints();
    d["max_hp"] = u->getType().maxHitPoints();
    d["shields"] = u->getShields();
    d["max_shields"] = u->getType().maxShields();
    d["energy"] = u->getEnergy();
    Player owner = u->getPlayer();
    d["owner"] = owner ? owner->getID() : -1;
    d["completed"] = u->isCompleted();
    d["flying"] = u->isFlying();
    d["idle"] = u->isIdle();
    d["moving"] = u->isMoving();
    d["attacking"] = u->isAttacking();
    d["under_attack"] = u->isUnderAttack();
    d["gw_cooldown"] = u->getGroundWeaponCooldown();
    d["aw_cooldown"] = u->getAirWeaponCooldown();
    d["vel_x"] = u->getVelocityX();
    d["vel_y"] = u->getVelocityY();
    d["order"] = u->getOrder().getID();
    return d;
}

std::vector<py::dict> collect_units(int player_id, int visible_to) {
    Player p = game().getPlayer(player_id);
    if (!p) throw std::runtime_error("no such player");
    Player viewer = visible_to >= 0 ? game().getPlayer(visible_to) : nullptr;
    std::vector<py::dict> out;
    for (Unit u : p->getUnits()) {
        if (!u->exists()) continue;
        if (viewer && !u->isVisible(viewer)) continue;
        out.push_back(unit_to_dict(u));
    }
    return out;
}

}  // namespace

PYBIND11_MODULE(_engine, m) {
    m.doc() = "FruitCraft bridge over OpenSnowstorm mini-openbwapi (one game per process)";

    m.def("update", []() {
        game().update();
        std::vector<py::dict> events;
        for (const Event& e : game().getEvents()) {
            py::dict d;
            d["type"] = (int)e.getType();
            const Event& ev = e;
            Unit u = ev.getUnit();
            d["unit_id"] = u ? u->getID() : -1;
            d["is_winner"] = ev.isWinner();
            events.push_back(d);
        }
        return events;
    }, "Advance one frame (or start/finish a match); returns events");

    m.def("in_game", []() { return game().isInGame(); });
    m.def("frame", []() { return game().getFrameCount(); });
    m.def("leave_game", []() { game().leaveGame(); });
    m.def("set_gui", [](bool enable) { game().setGUI(enable); });
    m.def("set_camera", [](int x, int y) { game().setScreenPosition(Position(x, y)); },
          "Top-left corner of the 800x600 view, map pixels (UI builds only)");
    m.def("set_random_seed", [](uint32_t seed) { game().setRandomSeed(seed); });
    m.def("disable_triggers", []() { game().disableTriggers(); });

    m.def("map_width", []() { return game().mapWidth(); });
    m.def("map_height", []() { return game().mapHeight(); });
    m.def("map_filename", []() { return game().mapFileName(); });
    m.def("start_locations", []() {
        std::vector<std::pair<int, int>> r;
        for (auto& v : game().getStartLocations()) r.emplace_back(v.x, v.y);
        return r;
    });

    m.def("self_player", []() { Player p = game().self(); return p ? p->getID() : -1; });
    m.def("enemy_player", []() { Player p = game().enemy(); return p ? p->getID() : -1; });
    m.def("player_info", [](int player_id) {
        Player p = game().getPlayer(player_id);
        if (!p) throw std::runtime_error("no such player");
        py::dict d;
        d["minerals"] = p->minerals();
        d["gas"] = p->gas();
        d["supply_used"] = p->supplyUsed();
        d["supply_total"] = p->supplyTotal();
        return d;
    });

    m.def("units", &collect_units,
          py::arg("player_id"), py::arg("visible_to") = -1,
          "Units of a player; visible_to filters by that player's fog of war");

    m.def("command", [](int unit_id, int type, int target_id, int x, int y, int extra) {
        if (!command_supported(type))
            throw std::runtime_error("unsupported command type " + std::to_string(type));
        Unit u = game().getUnit(unit_id);
        if (!u || !u->exists()) return false;
        Unit target = target_id >= 0 ? game().getUnit(target_id) : nullptr;
        return u->issueCommand(UnitCommand(u, UnitCommandType(type), target, x, y, extra));
    }, py::arg("unit_id"), py::arg("type"), py::arg("target_id") = -1,
       py::arg("x") = -1, py::arg("y") = -1, py::arg("extra") = 0);

    m.def("create_unit", [](int player_id, int unit_type, int x, int y) {
        Player p = game().getPlayer(player_id);
        if (!p) throw std::runtime_error("no such player");
        Unit u = game().createUnit(p, unit_type, Position(x, y));
        return u ? u->getID() : -1;
    });
    m.def("kill_unit", [](int unit_id) {
        Unit u = game().getUnit(unit_id);
        if (u && u->exists()) game().killUnit(u);
    });
    m.def("remove_unit", [](int unit_id) {
        Unit u = game().getUnit(unit_id);
        if (u && u->exists()) game().removeunit(u);
    });

    m.def("save_snapshot", [](const std::string& id) { game().saveSnapshot(id); });
    m.def("load_snapshot", [](const std::string& id) { game().loadSnapshot(id); });

    py::dict cmd;
    cmd["ATTACK_MOVE"] = (int)UnitCommandTypes::Attack_Move;
    cmd["ATTACK_UNIT"] = (int)UnitCommandTypes::Attack_Unit;
    cmd["MOVE"] = (int)UnitCommandTypes::Move;
    cmd["BUILD"] = (int)UnitCommandTypes::Build;
    cmd["TRAIN"] = (int)UnitCommandTypes::Train;
    cmd["RIGHT_CLICK_UNIT"] = (int)UnitCommandTypes::Right_Click_Unit;
    m.attr("COMMANDS") = cmd;

    py::dict ev;
    ev["MATCH_START"] = (int)EventType::MatchStart;
    ev["MATCH_END"] = (int)EventType::MatchEnd;
    ev["MATCH_FRAME"] = (int)EventType::MatchFrame;
    ev["UNIT_DESTROY"] = (int)EventType::UnitDestroy;
    m.attr("EVENTS") = ev;
}
