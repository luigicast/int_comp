#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==========================================================================
# analyze_metrics.py
#
# Parses the experiment logs produced by run_experiments.sh for a given maze,
# computes aggregate metrics per obstacle group, prints and saves a readable
# report (maze name, obstacle sets with their cells, training time, and all
# agreed metrics), writes per-run raw data to CSV, and saves plots:
#   - navigation time per group (bars, mean +/- sample std)
#   - replanifications per group (bars)
#   - physical vs compute time (stacked)
#   - overlaid trajectories per group with maze walls + mean path
#   - one individual trajectory plot per run
#
# Usage:
#   python analyze_metrics.py <maze>
#   e.g.  python analyze_metrics.py maze_06
# ==========================================================================

import os
import re
import sys
import csv
import json
import glob
import math

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---- Maze argument ------------------------------------------------------
if len(sys.argv) < 2:
    print("Usage: python analyze_metrics.py <maze>  (e.g. maze_06)")
    sys.exit(1)
MAZE_NAME = sys.argv[1]

HOME = os.path.expanduser('~')
ROS_DIR = os.path.join(HOME, 'ros_mazes')
METRIC_DIR = os.path.join(ROS_DIR, 'metrics', MAZE_NAME)
LOG_DIR = os.path.join(METRIC_DIR, 'logs')
TRAJ_DIR = os.path.join(METRIC_DIR, 'trajectories')
OUT_DIR = os.path.join(METRIC_DIR, 'results')
MAZE_CSV = os.path.join(ROS_DIR, 'csv', '{}.csv'.format(MAZE_NAME))
META_JSON = os.path.join(ROS_DIR, 'maps', '{}.json'.format(MAZE_NAME))
OBS_DIR = os.path.join(ROS_DIR, 'obstacles')

GROUPS = ['0obs', '1obs', '2obs']
GROUP_LABEL = {'0obs': '0 sets', '1obs': '1 set', '2obs': '2 sets'}

if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)

# ---- Regexes ------------------------------------------------------------
RE_SELECT  = re.compile(r'\[INFO\]\s*\[[\d.]+,\s*([\d.]+)\].*selected action')
RE_GOAL    = re.compile(r'\[INFO\]\s*\[[\d.]+,\s*([\d.]+)\].*Goal reached successfully')
RE_REPLAN  = re.compile(r'value iteration completed in ([\d.]+) ms')
RE_CONFIRM = re.compile(r'\[SCAN\] Obstacle confirmed at cell \((\d+),\s*(\d+)\)')
RE_MOVED   = re.compile(r'moved to gz=\(([-\d.]+),([-\d.]+)\)')
RE_ACTION  = re.compile(r'selected action (\d) \((\w+)\)')


# ---- Metadata: offsets, start/goal, obstacle sets, training time --------
def load_meta():
    res, ox, oy = 0.5, 0.0, 0.0
    start, goal = None, None
    if os.path.isfile(META_JSON):
        d = json.load(open(META_JSON))
        res = d.get('resolution', 0.5)
        ox = d.get('offset_x', 0.0)
        oy = d.get('offset_y', 0.0)
        start = tuple(d['start']['cell'])
        goal = tuple(d['goal']['cell'])
    return res, ox, oy, start, goal


def load_obstacle_cells(group):
    """Return the list of obstacle cells for a group, read from its JSON."""
    if group == '0obs':
        return []
    path = os.path.join(OBS_DIR, '{}_{}.json'.format(MAZE_NAME, group))
    if not os.path.isfile(path):
        return None
    d = json.load(open(path))
    return [tuple(c) for c in d.get('obstacles', [])]


def load_training_seconds():
    path = os.path.join(METRIC_DIR, 'training_seconds.txt')
    if os.path.isfile(path):
        try:
            return float(open(path).read().strip())
        except ValueError:
            return None
    return None


# ---- Parsing ------------------------------------------------------------
def parse_log(path):
    text = open(path).read()
    select_times = [float(t) for t in RE_SELECT.findall(text)]
    goal_match = RE_GOAL.search(text)
    replan_ms = [float(m) for m in RE_REPLAN.findall(text)]
    confirmed = set(RE_CONFIRM.findall(text))
    positions = [(float(x), float(y)) for (x, y) in RE_MOVED.findall(text)]
    actions = RE_ACTION.findall(text)
    success = goal_match is not None
    nav_time = None
    if success and select_times:
        nav_time = float(goal_match.group(1)) - select_times[0]
    n_diag = sum(1 for (a, _) in actions if int(a) % 2 == 1)
    detected_cells = set((int(r), int(c)) for (r, c) in confirmed)
    return {
        'success': success, 'nav_time': nav_time, 'n_steps': len(select_times),
        'n_replans': len(replan_ms), 'replan_ms': replan_ms,
        'n_obstacles': len(confirmed), 'n_diag': n_diag,
        'detected_cells': detected_cells,
        'positions': positions, 'traj': [],
    }


def load_trajectory_csv(group, run_idx):
    path = os.path.join(TRAJ_DIR, '{}_run{:02d}.csv'.format(group, run_idx))
    pts = []
    if not os.path.isfile(path):
        return pts
    with open(path) as f:
        next(f, None)
        for line in f:
            p = line.strip().split(',')
            if len(p) >= 3:
                try:
                    pts.append((float(p[1]), float(p[2])))
                except ValueError:
                    pass
    return pts


def collect():
    data = {g: [] for g in GROUPS}
    for g in GROUPS:
        paths = sorted(glob.glob(os.path.join(LOG_DIR, '{}_run*.log'.format(g))))
        for idx, path in enumerate(paths, 1):
            rec = parse_log(path)
            traj = load_trajectory_csv(g, idx)
            rec['traj'] = traj if traj else rec['positions']
            data[g].append(rec)
    return data


def mean_std(values):
    """Mean and SAMPLE std (ddof=1), ignoring None."""
    vals = [v for v in values if v is not None]
    if not vals:
        return (float('nan'), float('nan'), 0)
    arr = np.array(vals, dtype=float)
    s = float(arr.std(ddof=1)) if len(vals) > 1 else 0.0
    return (float(arr.mean()), s, len(vals))


def full_stats(values):
    """Return min, max, median, mean, sample std and count, ignoring None."""
    vals = [v for v in values if v is not None]
    if not vals:
        nan = float('nan')
        return {'min': nan, 'max': nan, 'median': nan, 'mean': nan, 'std': nan, 'n': 0}
    arr = np.array(vals, dtype=float)
    return {
        'min': float(arr.min()), 'max': float(arr.max()),
        'median': float(np.median(arr)), 'mean': float(arr.mean()),
        'std': float(arr.std(ddof=1)) if len(vals) > 1 else 0.0,
        'n': len(vals),
    }


def load_maze_walls(csv_path):
    grid = []
    with open(csv_path) as f:
        for line in f:
            line = line.strip()
            if line:
                grid.append([int(float(v)) for v in line.split(',')])
    return np.array(grid)


# ---- Report -------------------------------------------------------------
def write_report(data, meta, present_groups):
    res, ox, oy, start, goal = meta
    train_s = load_training_seconds()
    lines = []
    def out(s=''):
        lines.append(s)

    out("=" * 70)
    out("  NAVIGATION METRICS REPORT")
    out("=" * 70)
    out("  Maze:            {}".format(MAZE_NAME))
    if start and goal:
        out("  Start cell:      {}".format(start))
        out("  Goal cell:       {}".format(goal))
    out("  Resolution:      {} m/cell".format(res))
    out("  Runs per group:  10")
    if train_s is not None:
        out("  Training time:   {:.2f} s (one-time, per maze)".format(train_s))
    out("")
    out("  Obstacle sets used:")
    for g in present_groups:
        cells = load_obstacle_cells(g)
        if g == '0obs':
            out("    {} : none".format(GROUP_LABEL[g]))
        elif cells is None:
            out("    {} : (obstacle file not found)".format(GROUP_LABEL[g]))
        else:
            out("    {} : {} cells -> {}".format(GROUP_LABEL[g], len(cells), cells))
    out("")
    out("=" * 70)

    base_time = None
    prev_time = None
    for g in present_groups:
        runs = data[g]
        n = len(runs)
        if n == 0:
            out("\nGROUP {} ({}): no logs found".format(g, GROUP_LABEL[g]))
            continue
        succ = sum(1 for r in runs if r['success'])
        ok = [r for r in runs if r['success']]
        t_m, t_s, _ = mean_std([r['nav_time'] for r in ok])
        st_m, st_s, _ = mean_std([r['n_steps'] for r in ok])
        rp_m, rp_s, _ = mean_std([r['n_replans'] for r in ok])
        ob_m, ob_s, _ = mean_std([r['n_obstacles'] for r in ok])
        dg_m, dg_s, _ = mean_std([r['n_diag'] for r in ok])
        all_ms = [ms for r in ok for ms in r['replan_ms']]
        vi_m, vi_s, _ = mean_std(all_ms)

        out("")
        out("GROUP {} ({})".format(g, GROUP_LABEL[g]))
        out("  Success rate:           {}/{}  ({:.0f}%)".format(succ, n, 100.0 * succ / n))
        out("  Navigation time:        {:.1f} +/- {:.1f} s".format(t_m, t_s))
        if base_time is not None and not math.isnan(t_m):
            out("  Increase vs 0 sets:     {:+.1f}%".format(100.0 * (t_m - base_time) / base_time))
        if prev_time is not None and not math.isnan(t_m):
            out("  Increase vs previous:   {:+.1f}%".format(100.0 * (t_m - prev_time) / prev_time))
        # Extended stats for navigation time
        ns = full_stats([r['nav_time'] for r in ok])
        if ns['n'] > 0:
            out("    min / median / max:   {:.1f} / {:.1f} / {:.1f} s".format(
                ns['min'], ns['median'], ns['max']))
        out("  Steps (cells):          {:.1f} +/- {:.1f}".format(st_m, st_s))
        out("  Diagonal steps:         {:.1f} +/- {:.1f}".format(dg_m, dg_s))
        out("  Replanifications:       {:.1f} +/- {:.1f}".format(rp_m, rp_s))
        out("  Obstacle cells detected:{:.1f} +/- {:.1f}".format(ob_m, ob_s))
        if all_ms:
            out("  Compute per replan:     {:.1f} +/- {:.1f} ms".format(vi_m, vi_s))

        # Per-run breakdown: navigation time of every individual run
        out("  Per-run navigation time (s):")
        out("    run:  " + "  ".join("{:>2d}".format(i) for i in range(1, n + 1)))
        cells = []
        for r in runs:
            if r['success'] and r['nav_time'] is not None:
                cells.append("{:5.1f}".format(r['nav_time']))
            else:
                cells.append("  X  ")   # failed / no time
        out("    time: " + " ".join(cells))

        if base_time is None and not math.isnan(t_m):
            base_time = t_m
        if not math.isnan(t_m):
            prev_time = t_m

    # ---- False-positive detection (0obs group only) ----------------------
    # With no dynamic obstacles present, any replanification is triggered by a
    # spurious detection (sensor projection phantom). Flag the runs where it
    # happened so they are visible in the report.
    if '0obs' in present_groups:
        fp_runs = []
        for i, r in enumerate(data['0obs'], 1):
            if r['n_replans'] > 0:
                cells = sorted(r['detected_cells'])
                fp_runs.append((i, r['n_replans'], cells))
        out("")
        out("-" * 70)
        out("  FALSE POSITIVES (0 sets group)")
        if not fp_runs:
            out("    None. No run replanned without obstacles present (clean).")
        else:
            out("    {}/{} runs replanned with NO obstacles present -> phantom".format(
                len(fp_runs), len(data['0obs'])))
            for run_i, nrep, cells in fp_runs:
                out("      run {:02d}: {} replan(s), detected cell(s) {}".format(
                    run_i, nrep, cells if cells else "(none logged)"))
            out("    These are sensor-projection false positives (no real obstacle).")

    out("")
    out("  Units: navigation time in seconds (simulation clock); compute time in")
    out("  milliseconds (real wall clock); +/- is the sample standard deviation")
    out("  (N-1) across the 10 runs. Time/step metrics use successful runs only.")
    out("  'X' in per-run times marks a run that did not reach the goal.")
    out("=" * 70)

    report = "\n".join(lines)
    print(report)
    with open(os.path.join(OUT_DIR, 'summary_{}.txt'.format(MAZE_NAME)), 'w') as f:
        f.write(report + "\n")


def write_csv(data, present_groups):
    path = os.path.join(OUT_DIR, 'runs_{}.csv'.format(MAZE_NAME))
    with open(path, 'w') as f:
        w = csv.writer(f)
        w.writerow(['maze', 'group', 'run', 'success', 'nav_time_s', 'n_steps',
                    'n_replans', 'n_obstacle_cells', 'n_diag', 'mean_replan_ms'])
        for g in present_groups:
            for i, r in enumerate(data[g], 1):
                mrep = np.mean(r['replan_ms']) if r['replan_ms'] else ''
                w.writerow([MAZE_NAME, g, i, int(r['success']),
                            '' if r['nav_time'] is None else round(r['nav_time'], 2),
                            r['n_steps'], r['n_replans'], r['n_obstacles'],
                            r['n_diag'], '' if mrep == '' else round(mrep, 2)])
    print("Wrote {}".format(path))


def write_times_table(data, present_groups):
    """Write a wide table of per-run navigation times: one row per run, one column
    per obstacle group, plus Average and Std.Dev rows at the bottom. This mirrors
    the reference layout (Run | 0 obstacles | 1 obstacle | 2 obstacles ...)."""
    path = os.path.join(OUT_DIR, 'times_table_{}.csv'.format(MAZE_NAME))
    n_runs = max(len(data[g]) for g in present_groups)
    with open(path, 'w') as f:
        w = csv.writer(f)
        w.writerow(['Run'] + [GROUP_LABEL[g] for g in present_groups])
        for i in range(n_runs):
            row = [i + 1]
            for g in present_groups:
                runs = data[g]
                if i < len(runs) and runs[i]['success'] and runs[i]['nav_time'] is not None:
                    row.append(round(runs[i]['nav_time'], 3))
                else:
                    row.append('')   # failed / missing
            w.writerow(row)
        # Average and std rows (successful runs only)
        avg = ['Average']; sd = ['Std.Dev']
        for g in present_groups:
            ok = [r['nav_time'] for r in data[g] if r['success'] and r['nav_time'] is not None]
            m, s, _ = mean_std(ok)
            avg.append('' if math.isnan(m) else round(m, 3))
            sd.append('' if math.isnan(s) else round(s, 3))
        w.writerow(avg)
        w.writerow(sd)
    print("Wrote {}".format(path))


# ---- Plots --------------------------------------------------------------
def _gz_to_plot(pts, h, ox, oy, res):
    """Convert Gazebo (x, y) to plot (col, row) coordinates aligned with the maze
    drawn by imshow(origin='upper', extent=[0,w,0,h]). Mirrors the robot's own
    gz_to_cell mapping (spawn at 0) and places points like the policy plot does,
    so trajectories and policy share the same orientation (the row axis was
    previously flipped, drawing paths upside down)."""
    cols, rows = [], []
    for (x, y) in pts:
        col_frac = (x + ox - res / 2.0) / res
        row_frac = (h - 1) - (y + oy - res / 2.0) / res
        cols.append(col_frac + 0.5)
        rows.append(h - row_frac - 0.5)
    return cols, rows


def plot_time_bars(data, present_groups):
    means, stds, labels = [], [], []
    for g in present_groups:
        ok = [r['nav_time'] for r in data[g] if r['success'] and r['nav_time'] is not None]
        m, s, _ = mean_std(ok)
        means.append(m); stds.append(s); labels.append(GROUP_LABEL[g])
    x = np.arange(len(present_groups))
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x, means, yerr=stds, capsize=8, color='#3b7dd8', edgecolor='black')
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel('Navigation time (s)')
    ax.set_title('{}: navigation time by obstacle group'.format(MAZE_NAME))
    for i, m in enumerate(means):
        if not math.isnan(m):
            yy = m + (stds[i] if not math.isnan(stds[i]) else 0) + 1
            ax.text(i, yy, '{:.1f}s'.format(m), ha='center', fontsize=10)
    fig.tight_layout()
    p = os.path.join(OUT_DIR, 'time_by_group_{}.png'.format(MAZE_NAME))
    fig.savefig(p, dpi=130); plt.close(fig); print("Wrote {}".format(p))


def plot_replan_bars(data, present_groups):
    means, stds, labels = [], [], []
    for g in present_groups:
        ok = [r['n_replans'] for r in data[g] if r['success']]
        m, s, _ = mean_std(ok)
        means.append(m); stds.append(s); labels.append(GROUP_LABEL[g])
    x = np.arange(len(present_groups))
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x, means, yerr=stds, capsize=8, color='#e0902b', edgecolor='black')
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel('Replanifications')
    ax.set_title('{}: replanifications by obstacle group'.format(MAZE_NAME))
    fig.tight_layout()
    p = os.path.join(OUT_DIR, 'replans_by_group_{}.png'.format(MAZE_NAME))
    fig.savefig(p, dpi=130); plt.close(fig); print("Wrote {}".format(p))


def plot_phys_vs_compute(data, present_groups):
    phys, comp, labels = [], [], []
    for g in present_groups:
        ok = [r for r in data[g] if r['success'] and r['nav_time'] is not None]
        if not ok:
            phys.append(0); comp.append(0); labels.append(GROUP_LABEL[g]); continue
        nav = np.mean([r['nav_time'] for r in ok])
        comp_s = np.mean([sum(r['replan_ms']) / 1000.0 for r in ok])
        phys.append(nav - comp_s); comp.append(comp_s); labels.append(GROUP_LABEL[g])
    x = np.arange(len(present_groups))
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x, phys, label='Physical navigation', color='#3b7dd8', edgecolor='black')
    ax.bar(x, comp, bottom=phys, label='Compute (replan)', color='#d84b4b', edgecolor='black')
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel('Time (s)')
    ax.set_title('{}: physical navigation vs compute time'.format(MAZE_NAME))
    ax.legend()
    fig.tight_layout()
    p = os.path.join(OUT_DIR, 'phys_vs_compute_{}.png'.format(MAZE_NAME))
    fig.savefig(p, dpi=130); plt.close(fig); print("Wrote {}".format(p))


def plot_trajectories(data, walls, meta, present_groups):
    res, ox, oy, start, goal = meta
    h, w = walls.shape
    indiv_dir = os.path.join(OUT_DIR, 'trajectories_individual')
    if not os.path.exists(indiv_dir):
        os.makedirs(indiv_dir)

    # Distinct colors so the 10 individual runs are visible even when they nearly
    # overlap (the robot follows almost the same route every time).
    cmap = plt.get_cmap('tab10')

    for g in present_groups:
        runs = [r for r in data[g] if r['traj']]
        if not runs:
            continue
        obstacles = load_obstacle_cells(g) or []

        # Convert all runs once (reused by both plots)
        converted = []
        for r in runs:
            cols, rows = _gz_to_plot(r['traj'], h, ox, oy, res)
            converted.append((cols, rows))

        # ---- Plot A: the 10 individual runs overlaid (no mean) ----
        fig, ax = plt.subplots(figsize=(7.5, 7.5))
        _draw_maze_background(ax, walls, obstacles, start, goal, h, w)
        for idx, (cols, rows) in enumerate(converted):
            ax.plot(cols, rows, color=cmap(idx % 10), alpha=0.8, linewidth=1.6,
                    label='run {:02d}'.format(idx + 1))
        ax.set_xlim(0, w); ax.set_ylim(0, h)
        _row_axis_labels(ax, h, w)
        ax.set_title('{} - {} ({} runs overlaid)'.format(MAZE_NAME, GROUP_LABEL[g], len(runs)))
        ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=7)
        ax.set_xlabel('column')
        fig.tight_layout()
        p = os.path.join(OUT_DIR, 'trajectories_{}_{}.png'.format(MAZE_NAME, g))
        fig.savefig(p, dpi=130); plt.close(fig); print("Wrote {}".format(p))

        # ---- Plot B: the mean path only ----
        L = min(len(c) for c, _ in converted)
        if L >= 2:
            mc = np.mean([np.interp(np.linspace(0, 1, L), np.linspace(0, 1, len(c)), c)
                          for c, _ in converted], axis=0)
            mr = np.mean([np.interp(np.linspace(0, 1, L), np.linspace(0, 1, len(rw)), rw)
                          for _, rw in converted], axis=0)
            fig, ax = plt.subplots(figsize=(7.5, 7.5))
            _draw_maze_background(ax, walls, obstacles, start, goal, h, w)
            ax.plot(mc, mr, color='black', linewidth=3.5, label='Mean path', zorder=10)
            ax.set_xlim(0, w); ax.set_ylim(0, h)
            _row_axis_labels(ax, h, w)
            ax.set_title('{} - {} (mean path)'.format(MAZE_NAME, GROUP_LABEL[g]))
            ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=8)
            ax.set_xlabel('column')
            fig.tight_layout()
            p = os.path.join(OUT_DIR, 'trajectories_{}_{}_mean.png'.format(MAZE_NAME, g))
            fig.savefig(p, dpi=130); plt.close(fig); print("Wrote {}".format(p))

        for i, r in enumerate(runs, 1):
            fig, ax = plt.subplots(figsize=(6.5, 6.5))
            _draw_maze_background(ax, walls, obstacles, start, goal, h, w)
            cols, rows = _gz_to_plot(r['traj'], h, ox, oy, res)
            color = '#1e8c3a' if r['success'] else '#d81e1e'
            ax.plot(cols, rows, color=color, linewidth=2)
            if cols:
                ax.plot(cols[0], rows[0], 'o', color='black', markersize=7, label='start')
                ax.plot(cols[-1], rows[-1], '*', color=color, markersize=14,
                        label='goal' if r['success'] else 'end (fail)')
            ax.set_xlim(0, w); ax.set_ylim(0, h)
            _row_axis_labels(ax, h, w)
            status = 'OK' if r['success'] else 'FAIL'
            ax.set_title('{} {} run {:02d} [{}]'.format(MAZE_NAME, GROUP_LABEL[g], i, status))
            ax.legend(loc='upper right', fontsize=8)
            fig.tight_layout()
            p = os.path.join(indiv_dir, '{}_{}_run{:02d}.png'.format(MAZE_NAME, g, i))
            fig.savefig(p, dpi=110); plt.close(fig)
        print("Wrote {} individual plots for {}".format(len(runs), g))


def _row_axis_labels(ax, h, w=None):
    """Relabel the Y axis so it reads as CSV rows: row 0 at the top, increasing
    downward. Shows every row tick (0..h-1), and every column tick too, so each
    cell is directly identifiable. Only tick LABELS change, not the drawing, so
    trajectories and walls stay exactly as plotted. Plot y-value Y = CSV row (h-Y)."""
    yticks = list(range(0, h + 1))
    ax.set_yticks(yticks)
    ax.set_yticklabels([str(h - t) for t in yticks])
    ax.set_ylabel('row (CSV order: 0 at top)')
    if w is not None:
        ax.set_xticks(list(range(0, w + 1)))
        ax.set_xticklabels([str(c) for c in range(0, w + 1)])


def _draw_maze_background(ax, walls, obstacles, start, goal, h, w):
    """Draw the maze: static walls (grey), a cell grid, dynamic obstacles (orange),
    and the start/goal markers. Shared by overlay and individual trajectory plots."""
    # Static walls
    ax.imshow(walls, cmap='Greys', origin='upper',
              extent=[0, w, 0, h], alpha=0.85, aspect='equal')

    # Dynamic obstacles as orange squares (one per cell)
    for (r, c) in obstacles:
        ax.add_patch(plt.Rectangle((c, h - r - 1), 1, 1,
                                   facecolor='#e8821e', edgecolor='#a35a10',
                                   alpha=0.9, zorder=3))

    # Cell grid lines
    for x in range(w + 1):
        ax.axvline(x, color='#b0b0b0', linewidth=0.5, alpha=0.6, zorder=1)
    for y in range(h + 1):
        ax.axhline(y, color='#b0b0b0', linewidth=0.5, alpha=0.6, zorder=1)

    # Start (S) and goal (G) markers
    if start:
        sr, sc = start
        ax.text(sc + 0.5, h - sr - 0.5, 'S', ha='center', va='center',
                fontsize=13, fontweight='bold', color='#d81e1e', zorder=5)
    if goal:
        gr, gc = goal
        ax.text(gc + 0.5, h - gr - 0.5, 'G', ha='center', va='center',
                fontsize=13, fontweight='bold', color='#1e8c3a', zorder=5)


def plot_base_policy(walls, meta):
    """Draw the trained base policy (no dynamic obstacles) as arrows over the
    maze. Reads the saved Q-table .npy for this maze. One arrow per free cell
    showing the best action; clockwise order 0=N 1=NE 2=E 3=SE 4=S 5=SW 6=W 7=NW."""
    res, ox, oy, start, goal = meta
    qpath = os.path.join(ROS_DIR, 'maps', '{}_q_table.npy'.format(MAZE_NAME))
    if not os.path.isfile(qpath):
        print("Note: Q-table not found ({}), skipping policy plot".format(qpath))
        return
    q = np.load(qpath)
    h, w = walls.shape

    # Unit direction (dx, dy) for each action, in plotting coords (col, row).
    # Row increases downward in the grid; we plot with row axis flipped, so a
    # "north" move (row-1) points up (+ in plot y). Arrow vectors below are in
    # (col_component, row_plot_component).
    # clockwise: N, NE, E, SE, S, SW, W, NW
    dirs = {
        0: (0, 1), 1: (1, 1), 2: (1, 0), 3: (1, -1),
        4: (0, -1), 5: (-1, -1), 6: (-1, 0), 7: (-1, 1),
    }

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(walls, cmap='Greys', origin='upper',
              extent=[0, w, 0, h], alpha=0.85, aspect='equal')

    for r in range(min(h, q.shape[0])):
        for c in range(min(w, q.shape[1])):
            if walls[r, c] == 1:
                continue
            if goal and (r, c) == tuple(goal):
                ax.text(c + 0.5, h - r - 0.5, 'G', ha='center', va='center',
                        fontsize=12, fontweight='bold', color='#1e8c3a')
                continue
            best = int(np.argmax(q[r, c]))
            dx, dy = dirs[best]
            # center of the cell in plot coords
            cx = c + 0.5
            cy = h - r - 0.5
            ax.arrow(cx - dx * 0.18, cy - dy * 0.18, dx * 0.3, dy * 0.3,
                     head_width=0.16, head_length=0.14,
                     fc='#3b7dd8', ec='#3b7dd8', length_includes_head=True)

    if start:
        sr, sc = start
        ax.text(sc + 0.5, h - sr - 0.5, 'S', ha='center', va='center',
                fontsize=12, fontweight='bold', color='#d81e1e')

    ax.set_xlim(0, w); ax.set_ylim(0, h)
    _row_axis_labels(ax, h, w)
    ax.set_title('{}: base policy (no dynamic obstacles)'.format(MAZE_NAME))
    ax.set_xlabel('column'); ax.set_ylabel('row')
    fig.tight_layout()
    p = os.path.join(OUT_DIR, 'policy_base_{}.png'.format(MAZE_NAME))
    fig.savefig(p, dpi=130); plt.close(fig)
    print("Wrote {}".format(p))


def main():
    if not os.path.isfile(MAZE_CSV):
        print("Maze CSV not found: {}".format(MAZE_CSV)); return
    meta = load_meta()
    walls = load_maze_walls(MAZE_CSV)
    data = collect()

    # Only report groups that actually have logs
    present = [g for g in GROUPS if data[g]]
    if not present:
        print("No logs found in {}. Run run_experiments.sh first.".format(LOG_DIR)); return

    write_report(data, meta, present)
    write_csv(data, present)
    write_times_table(data, present)
    plot_time_bars(data, present)
    plot_replan_bars(data, present)
    plot_phys_vs_compute(data, present)
    plot_trajectories(data, walls, meta, present)
    plot_base_policy(walls, meta)
    print("\nAll results saved to {}".format(OUT_DIR))


if __name__ == '__main__':
    main()
