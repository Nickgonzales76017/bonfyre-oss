#!/usr/bin/env bash
set -euo pipefail

# Spec-driven E2E runner
# Reads scripts/e2e_tests.json and executes each test's commands
# Captures stdout/stderr to site/demos/e2e-results/<test>/ and evaluates expectations

SPEC_FILE="${1:-scripts/e2e_tests.json}"
OUT_DIR="site/demos/e2e-results"
mkdir -p "$OUT_DIR"

if ! command -v jq >/dev/null 2>&1; then
  echo "jq is required. Install it first." >&2
  exit 2
fi

tests_count=$(jq '.tests | length' "$SPEC_FILE")
echo "Loaded $tests_count tests from $SPEC_FILE"

PASS=0
FAIL=0
results=()

for i in $(seq 0 $((tests_count-1))); do
  name=$(jq -r ".tests[$i].name" "$SPEC_FILE")
  echo "== Test: $name =="
  test_out="$OUT_DIR/$name"
  mkdir -p "$test_out"

  # read commands as full lines into an array to preserve spacing/ && sequences
  cmds_arr=()
  while IFS= read -r line; do
    cmds_arr+=("$line")
  done < <(jq -r ".tests[$i].commands[]?" "$SPEC_FILE")
  rc_expected=$(jq -r ".tests[$i].expect.rc // empty" "$SPEC_FILE" || true)
  stdout_json=$(jq -r ".tests[$i].expect.stdout_json // false" "$SPEC_FILE")
  stdout_json_path=$(jq -r ".tests[$i].expect.stdout_json_path // empty" "$SPEC_FILE")
  min_composite=$(jq -r ".tests[$i].expect.min_composite // empty" "$SPEC_FILE")
  expect_files=$(jq -r ".tests[$i].expect.files // [] | @sh" "$SPEC_FILE")
  expect_json_paths=$(jq -r ".tests[$i].expect.json_paths // [] | @sh" "$SPEC_FILE")

  test_rc=0
  combined_stdout="$test_out/combined.stdout"
  : > "$combined_stdout"

  for cmd in "${cmds_arr[@]}"; do
    echo "+ $cmd" | tee -a "$test_out/summary.txt"
    set +e
    bash -lc "$cmd" >> "$combined_stdout" 2> "$test_out/cmd.stderr"
    cmd_rc=$?
    set -e
    if [ $cmd_rc -ne 0 ]; then
      echo "command failed with rc=$cmd_rc" | tee -a "$test_out/summary.txt"
      test_rc=$cmd_rc
      # continue to collect remaining commands/artifacts rather than break
    fi
  done

  # evaluate expectations
  passed=true
  if [ -n "$rc_expected" ]; then
    if [ "$test_rc" -ne "$rc_expected" ]; then
      passed=false
      echo "rc mismatch: got $test_rc expected $rc_expected" | tee -a "$test_out/summary.txt"
    fi
  fi

  if [ "$stdout_json" = "true" ]; then
    # try to parse combined stdout as JSON and extract path
    comp=$(jq -r "$stdout_json_path // empty" < "$combined_stdout" 2>/dev/null || true)
    if [ -z "$comp" ]; then
      passed=false
      echo "failed to extract $stdout_json_path from stdout" | tee -a "$test_out/summary.txt"
    else
      echo "extracted $stdout_json_path = $comp" | tee -a "$test_out/summary.txt"
      if [ -n "$min_composite" ]; then
        # compare floating point numbers
        awk -v a=$comp -v b=$min_composite 'BEGIN{if (a+0 < b+0) exit 1; exit 0}'
        if [ $? -ne 0 ]; then
          passed=false
          echo "composite $comp < min_composite $min_composite" | tee -a "$test_out/summary.txt"
        fi
      fi
    fi
  fi

  # check expected files
  if [ -n "$expect_files" ] && [ "$expect_files" != "''" ]; then
    # shell-eval the @sh representation to get an array
    eval "files_arr=($expect_files)"
    for f in "${files_arr[@]}"; do
      if [ ! -e "$f" ]; then
        passed=false
        echo "expected file missing: $f" | tee -a "$test_out/summary.txt"
      else
        echo "found expected file: $f" | tee -a "$test_out/summary.txt"
      fi
    done
  fi

  # check expected json paths within files
  if [ -n "$expect_json_paths" ] && [ "$expect_json_paths" != "''" ]; then
    eval "json_arr=($expect_json_paths)"
    for jp in "${json_arr[@]}"; do
      # jp expected format: file:path
      file=$(echo "$jp" | cut -d':' -f1)
      path=$(echo "$jp" | cut -d':' -f2-)
      if [ ! -f "$file" ]; then
        passed=false
        echo "expected json file missing: $file" | tee -a "$test_out/summary.txt"
      else
        val=$(jq -r "$path // empty" < "$file" 2>/dev/null || true)
        if [ -z "$val" ]; then
          passed=false
          echo "json path $path empty in $file" | tee -a "$test_out/summary.txt"
        else
          echo "json path $path present in $file -> $val" | tee -a "$test_out/summary.txt"
        fi
      fi
    done
  fi

  if [ "$passed" = true ]; then
    echo "$name: PASS" | tee -a "$test_out/summary.txt"
    results+=("$name:PASS")
    PASS=$((PASS+1))
  else
    echo "$name: FAIL" | tee -a "$test_out/summary.txt"
    results+=("$name:FAIL")
    FAIL=$((FAIL+1))
  fi

done

echo
echo "Spec E2E Summary: PASS=$PASS FAIL=$FAIL (details in $OUT_DIR)"
for r in "${results[@]}"; do echo " - $r"; done

if [ $FAIL -gt 0 ]; then
  exit 1
fi

echo "All spec tests passed." 
