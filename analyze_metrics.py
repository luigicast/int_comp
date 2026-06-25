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
    return {
        'success': success, 'nav_time': nav_time, 'n_steps': len(select_times),
        'n_replans': len(replan_ms), 'replan_ms': replan_ms,
        'n_obstacles': len(confirmed), 'n_diag': n_diag,
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
        out("  Steps (cells):          {:.1f} +/- {:.1f}".format(st_m, st_s))
        out("  Diagonal steps:         {:.1f} +/- {:.1f}".format(dg_m, dg_s))
        out("  Replanifications:       {:.1f} +/- {:.1f}".format(rp_m, rp_s))
        out("  Obstacle cells detected:{:.1f} +/- {:.1f}".format(ob_m, ob_s))
        if all_ms:
            out("  Compute per replan:     {:.1f} +/- {:.1f} ms".format(vi_m, vi_s))
        if base_time is None and not math.isnan(t_m):
            base_time = t_m
        if not math.isnan(t_m):
            prev_time = t_m

    out("")
    out("  Units: navigation time in seconds (simulation clock); compute time in")
    out("  milliseconds (real wall clock); +/- is the sample standard deviation")
    out("  (N-1) across the 10 runs. Time/step metrics use successful runs only.")
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


# ---- Plots --------------------------------------------------------------
def _gz_to_plot(pts, h, ox, oy, res):
    cols = [(x + ox) / res for (x, y) in pts]
    rows = [h - ((y + oy) / res) for (x, y) in pts]
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

    for g in present_groups:
        runs = [r for r in data[g] if r['traj']]
        if not runs:
            continue
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.imshow(walls, cmap='Greys', origin='upper',
                  extent=[0, w, 0, h], alpha=0.85, aspect='equal')
        converted = []
        for r in runs:
            cols, rows = _gz_to_plot(r['traj'], h, ox, oy, res)
            ax.plot(cols, rows, color='#3b7dd8', alpha=0.30, linewidth=1.5)
            converted.append((cols, rows))
        L = min(len(c) for c, _ in converted)
        if L >= 2:
            mc = np.mean([np.interp(np.linspace(0, 1, L), np.linspace(0, 1, len(c)), c)
                          for c, _ in converted], axis=0)
            mr = np.mean([np.interp(np.linspace(0, 1, L), np.linspace(0, 1, len(rw)), rw)
                          for _, rw in converted], axis=0)
            ax.plot(mc, mr, color='#d81e1e', linewidth=3, label='Mean path')
        ax.set_xlim(0, w); ax.set_ylim(0, h)
        ax.set_title('{} - {} ({} runs)'.format(MAZE_NAME, GROUP_LABEL[g], len(runs)))
        ax.legend(loc='upper right')
        ax.set_xlabel('column'); ax.set_ylabel('row')
        fig.tight_layout()
        p = os.path.join(OUT_DIR, 'trajectories_{}_{}.png'.format(MAZE_NAME, g))
        fig.savefig(p, dpi=130); plt.close(fig); print("Wrote {}".format(p))

        for i, r in enumerate(runs, 1):
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(walls, cmap='Greys', origin='upper',
                      extent=[0, w, 0, h], alpha=0.85, aspect='equal')
            cols, rows = _gz_to_plot(r['traj'], h, ox, oy, res)
            color = '#1e8c3a' if r['success'] else '#d81e1e'
            ax.plot(cols, rows, color=color, linewidth=2)
            if cols:
                ax.plot(cols[0], rows[0], 'o', color='black', markersize=7, label='start')
                ax.plot(cols[-1], rows[-1], '*', color=color, markersize=14,
                        label='goal' if r['success'] else 'end (fail)')
            ax.set_xlim(0, w); ax.set_ylim(0, h)
            status = 'OK' if r['success'] else 'FAIL'
            ax.set_title('{} {} run {:02d} [{}]'.format(MAZE_NAME, GROUP_LABEL[g], i, status))
            ax.legend(loc='upper right', fontsize=8)
            fig.tight_layout()
            p = os.path.join(indiv_dir, '{}_{}_run{:02d}.png'.format(MAZE_NAME, g, i))
            fig.savefig(p, dpi=110); plt.close(fig)
        print("Wrote {} individual plots for {}".format(len(runs), g))


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
    plot_time_bars(data, present)
    plot_replan_bars(data, present)
    plot_phys_vs_compute(data, present)
    plot_trajectories(data, walls, meta, present)
    print("\nAll results saved to {}".format(OUT_DIR))


if __name__ == '__main__':
    main()
