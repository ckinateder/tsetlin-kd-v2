#!/bin/bash
# Migration script for renaming conventions in tsetlin-kd-v2
# Converts: student→baseline, distilled→student, distilled_ds→student_ds
#
# IMPORTANT: Uses temporary placeholders to avoid double-conversion
# Order: distilled_ds→TEMP1, distilled→TEMP2, student→baseline, TEMP1→student_ds, TEMP2→student

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="$REPO_ROOT/results"

# Temporary placeholders (unlikely to appear in real data)
TEMP_STUDENT="__MIGRATE_STUDENT_PLACEHOLDER__"
TEMP_STUDENT_DS="__MIGRATE_STUDENT_DS_PLACEHOLDER__"

# Cross-platform sed -i (macOS vs Linux)
sedi() {
    if [[ "$OSTYPE" == "darwin"* ]]; then
        sed -i '' "$@"
    else
        sed -i "$@"
    fi
}

echo "=== Tsetlin KD Naming Convention Migration ==="
echo "Repository root: $REPO_ROOT"
echo "Results directory: $RESULTS_DIR"
echo ""

# Check if results directory exists
if [ ! -d "$RESULTS_DIR" ]; then
    echo "No results directory found. Nothing to migrate."
    exit 0
fi

# Create backup
BACKUP_DIR="$REPO_ROOT/results_backup_$(date +%Y%m%d_%H%M%S)"
echo "Creating backup at: $BACKUP_DIR"
cp -r "$RESULTS_DIR" "$BACKUP_DIR"
echo "Backup created successfully."
echo ""

# Function to migrate a single file
migrate_file() {
    local file="$1"
    local ext="${file##*.}"

    if [[ "$ext" == "json" ]] || [[ "$ext" == "csv" ]]; then
        # Step 1: distilled_ds → temporary placeholder
        sedi "s/distilled_ds/$TEMP_STUDENT_DS/g" "$file"

        # Step 2: distilled → temporary placeholder
        sedi "s/distilled/$TEMP_STUDENT/g" "$file"

        # Step 3: student → baseline (but NOT in "student": { params block)
        # We need to be selective - only change metric keys, not params
        # Metric keys have patterns like: acc_test_student, time_train_student, etc.
        sedi 's/acc_test_student/acc_test_baseline/g' "$file"
        sedi 's/acc_train_student/acc_train_baseline/g' "$file"
        sedi 's/time_train_student/time_train_baseline/g' "$file"
        sedi 's/time_test_student/time_test_baseline/g' "$file"
        sedi 's/inference_time_student/inference_time_baseline/g' "$file"
        sedi 's/sum_time_train_student/sum_time_train_baseline/g' "$file"
        sedi 's/sum_time_test_student/sum_time_test_baseline/g' "$file"
        sedi 's/final_acc_test_student/final_acc_test_baseline/g' "$file"
        sedi 's/final_acc_train_student/final_acc_train_baseline/g' "$file"

        # Step 4: Restore temporary placeholders to final names
        sedi "s/$TEMP_STUDENT_DS/student_ds/g" "$file"
        sedi "s/$TEMP_STUDENT/student/g" "$file"

        echo "  Migrated: $file"
    fi
}

# Function to rename PKL files
rename_pkl_files() {
    local dir="$1"

    # Rename student_baseline.pkl → baseline.pkl
    if [ -f "$dir/student_baseline.pkl" ] && [ ! -f "$dir/baseline.pkl" ]; then
        mv "$dir/student_baseline.pkl" "$dir/baseline.pkl"
        echo "  Renamed: student_baseline.pkl → baseline.pkl"
    fi

    # Rename distilled.pkl → student.pkl
    if [ -f "$dir/distilled.pkl" ] && [ ! -f "$dir/student.pkl" ]; then
        mv "$dir/distilled.pkl" "$dir/student.pkl"
        echo "  Renamed: distilled.pkl → student.pkl"
    fi

    # Rename distilled_ds.pkl → student_ds.pkl
    if [ -f "$dir/distilled_ds.pkl" ] && [ ! -f "$dir/student_ds.pkl" ]; then
        mv "$dir/distilled_ds.pkl" "$dir/student_ds.pkl"
        echo "  Renamed: distilled_ds.pkl → student_ds.pkl"
    fi
}

# Process all directories recursively
echo "Processing files in $RESULTS_DIR..."
echo ""

# Find and process all JSON and CSV files
find "$RESULTS_DIR" -type f \( -name "*.json" -o -name "*.csv" \) | while read -r file; do
    migrate_file "$file"
done

# Find and rename PKL files in all directories
find "$RESULTS_DIR" -type d | while read -r dir; do
    rename_pkl_files "$dir"
done

echo ""
echo "=== Migration Complete ==="
echo "Backup saved at: $BACKUP_DIR"
echo ""
echo "Summary of changes:"
echo "  - JSON keys: *_student → *_baseline, *_distilled → *_student"
echo "  - CSV columns: same pattern"
echo "  - PKL files: student_baseline.pkl → baseline.pkl, distilled.pkl → student.pkl"
echo ""
echo "To verify, check a sample file:"
echo "  cat $RESULTS_DIR/aggregate_distribution/MNIST/aggregated_output.json | grep -E 'acc_test_(baseline|student)'"
