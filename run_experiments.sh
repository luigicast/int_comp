#!/bin/bash
# ==========================================================================
# run_experiments.sh
#
# Runs the navigation test N times for each obstacle group on a chosen maze,
# relaunching Gazebo each run for a clean state. Reads the start/goal from the
# maze's metadata JSON (produced earlier by csv_to_ros_map.py), measures the
# training time once, and saves a log and a trajectory CSV per run.
#
# Usage:
#   ./run_experiments.sh <maze> <max_obstacle_sets>
#
#   <maze>               one of: maze_02 maze_03 maze_06 maze_07
#   <max_obstacle_sets>  1 or 2  (1 -> groups 0,1 ; 2 -> groups 0,1,2)
#
# Example:
#   ./run_experiments.sh maze_06 2
#   ./run_experiments.sh maze_03 1
#
# Obstacle set JSON files (you create these) must live in:
#   ~/ros_mazes/obstacles/<maze>_1obs.json
#   ~/ros_mazes/obstacles/<maze>_2obs.json
# and contain {"obstacles": [[r,c],...]}. The 0-set group needs no file.
#
# After this finishes, run:  python analyze_metrics.py <maze>
# ==========================================================================

set -u

# ---- Arguments ----------------------------------------------------------
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <maze> <max_obstacle_sets(1|2)>"
    echo "  maze in: maze_02 maze_03 maze_06 maze_07"
    exit 1
fi

MAZE_NAME="$1"
MAX_OBS="$2"

case "$MAZE_NAME" in
    maze_02|maze_03|maze_06|maze_07) ;;
    *) echo "Error: maze must be one of maze_02 maze_03 maze_06 maze_07"; exit 1 ;;
esac

if [ "$MAX_OBS" != "1" ] && [ "$MAX_OBS" != "2" ]; then
    echo "Error: max_obstacle_sets must be 1 or 2"
    exit 1
fi

# ---- Configuration ------------------------------------------------------
ROS_DIR=~/ros_mazes
MAZE=$ROS_DIR/csv/${MAZE_NAME}.csv
META_JSON=$ROS_DIR/maps/${MAZE_NAME}.json
RESOLUTION=0.5
RUNS=10
GAZEBO_BOOT_WAIT=40
RUN_TIMEOUT=300

OBS_DIR=$ROS_DIR/obstacles
WORLD=$ROS_DIR/worlds/${MAZE_NAME}.world
GEN=~/csv_to_ros_map.py
TESTER=$ROS_DIR/q_learning_maze.py

OUT_DIR=$ROS_DIR/metrics/${MAZE_NAME}/logs
TRAJ_DIR=$ROS_DIR/metrics/${MAZE_NAME}/trajectories
META_DIR=$ROS_DIR/metrics/${MAZE_NAME}
mkdir -p "$OUT_DIR" "$TRAJ_DIR"

# ---- Read start/goal from the maze metadata JSON ------------------------
if [ ! -f "$META_JSON" ]; then
    echo "Error: metadata JSON not found: $META_JSON"
    echo "Generate it first with csv_to_ros_map.py (it writes start/goal)."
    exit 1
fi

read START_ROW START_COL GOAL_ROW GOAL_COL << EOF
$(python -c "
import json
d = json.load(open('$META_JSON'))
s = d['start']['cell']; g = d['goal']['cell']
print(s[0], s[1], g[0], g[1])
")
EOF

START="$START_ROW $START_COL"
GOAL="$GOAL_ROW $GOAL_COL"
echo "Maze $MAZE_NAME : start ($START)  goal ($GOAL)"

# ---- Build the list of groups to run ------------------------------------
GROUPS="0obs 1obs"
if [ "$MAX_OBS" = "2" ]; then
    GROUPS="0obs 1obs 2obs"
fi

# ---- Helpers ------------------------------------------------------------
kill_gazebo() {
    killall -9 gzserver gzclient roslaunch >/dev/null 2>&1
    sleep 2
}

launch_gazebo() {
    export TURTLEBOT_GAZEBO_WORLD_FILE="$WORLD"
    roslaunch turtlebot_gazebo turtlebot_world.launch >/dev/null 2>&1 &
    sleep "$GAZEBO_BOOT_WAIT"
}

obstacle_json_for() {
    case "$1" in
        0obs) echo "" ;;
        1obs) echo "$OBS_DIR/${MAZE_NAME}_1obs.json" ;;
        2obs) echo "$OBS_DIR/${MAZE_NAME}_2obs.json" ;;
    esac
}

# ---- Measure training time once (per maze) ------------------------------
echo "=============================================================="
echo "Measuring training time for $MAZE_NAME ..."
echo "=============================================================="
TRAIN_LOG="$META_DIR/training.log"
train_start=$(date +%s.%N)
python "$TESTER" --train --maze "$MAZE" > "$TRAIN_LOG" 2>&1
train_end=$(date +%s.%N)
TRAIN_SECONDS=$(python -c "print('%.2f' % ($train_end - $train_start))")
echo "$TRAIN_SECONDS" > "$META_DIR/training_seconds.txt"
echo "Training took $TRAIN_SECONDS s (saved to training_seconds.txt)"

# ---- Main loop ----------------------------------------------------------
for group in $GROUPS; do
    json="$(obstacle_json_for "$group")"
    echo "=============================================================="
    echo "GROUP $group   maze=$MAZE_NAME"
    if [ -n "$json" ]; then
        echo "  obstacles: $json"
        if [ ! -f "$json" ]; then
            echo "  Error: obstacle file missing: $json"
            echo "  Create it ({\"obstacles\": [[r,c],...]}) and re-run."
            exit 1
        fi
    else
        echo "  obstacles: none"
    fi
    echo "=============================================================="

    if [ -n "$json" ]; then
        python "$GEN" "$MAZE" "$RESOLUTION" --start $START --goal $GOAL --dynamic "$json"
    else
        python "$GEN" "$MAZE" "$RESOLUTION" --start $START --goal $GOAL
    fi

    for i in $(seq 1 "$RUNS"); do
        run_id=$(printf "%02d" "$i")
        log="$OUT_DIR/${group}_run${run_id}.log"
        export TRAJECTORY_FILE="$TRAJ_DIR/${group}_run${run_id}.csv"
        echo "--- $group run $run_id -> $log"

        kill_gazebo
        launch_gazebo

        timeout "$RUN_TIMEOUT" python "$TESTER" --test --maze "$MAZE" > "$log" 2>&1
        echo "    done (exit $?)"
    done
done

kill_gazebo
echo ""
echo "All runs finished for $MAZE_NAME. Logs in $OUT_DIR"

# ---- Automatically generate the report and plots ------------------------
ANALYZER=~/analyze_metrics.py
if [ -f "$ANALYZER" ]; then
    echo "=============================================================="
    echo "Generating metrics report and plots for $MAZE_NAME ..."
    echo "=============================================================="
    python3 "$ANALYZER" "$MAZE_NAME" || python "$ANALYZER" "$MAZE_NAME"
else
    echo "Note: analyze_metrics.py not found at $ANALYZER"
    echo "Run it manually:  python analyze_metrics.py $MAZE_NAME"
fi
