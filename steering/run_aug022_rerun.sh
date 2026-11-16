#!/usr/bin/env bash
# Rerun every CWE-022 setting in report tables 1 and 2 with the augmented safety
# vector (qwen25-coder-7b_intra_augspec022). Generation only; CodeQL / judge /
# summarize / audit are run separately afterwards.
#
# Usage (inside the container, from /workspace):
#   bash steering/run_aug022_rerun.sh
#   STEPS="5 6" bash steering/run_aug022_rerun.sh   # run a subset of steps
#
# Steps -> output mode directory under $OUT:
#   1 aug022_cmd_A       CausalMD A track (probe top-256), k16 a+5, all positions
#   2 aug022_cmd         CausalMD B track, k4 / k64, a+3, all positions
#   3 aug022_cmd_resp    CausalMD B track, k16 a+5, response-only
#   4 aug022_union_resp  safe 64 + vuln 4 heads, a+3, response-only (CausalMD+)
#   5 aug022_duo_A       DuoSteer A track, 16/16, +5/+5, all positions
#   6 aug022_duo_B       DuoSteer+ B track, 64/16, +3/-3, all positions
# Each step logs to logs/aug022_<mode>.log. steer_eval / duosteer_eval skip task ids
# already written to the output file, so rerunning resumes.

MODEL=Qwen/Qwen2.5-Coder-7B-Instruct
SLUG=Qwen_Qwen2_5-Coder-7B-Instruct
OUT=results/qwen/steering
CWE=cwe-022
TASKS=data/eval_tasks/seccodeplt_cwe022.jsonl
AUGSPEC=data/steering_vectors/$SLUG/qwen25-coder-7b_intra_augspec022/cwe-022/response_mean/head/mean_diff
SCAUSAL=data/causal_analysis/$SLUG/cwe-022/head_causal_results
CCAUSAL=data/causal_analysis_correctness/$SLUG/cwe-022/head_causal_results
CSV=data/steering_vectors_correctness/$SLUG/cwe-022_train/cwe-022/response_mean/head/mean_diff
STEPS=${STEPS:-1 2 3 4 5 6}
MODES="aug022_cmd_A aug022_cmd aug022_cmd_resp aug022_union_resp aug022_duo_A aug022_duo_B"

mkdir -p logs

cmd() {  # usage: cmd <mode> <ranking.json> <positions> <alpha> <top_k list> [vuln_k]
  python steering/steer_eval.py --setting head \
      --input_file $TASKS --model $MODEL --output_dir $OUT/$1/$CWE \
      --steer_positions $3 --head_results $2 --head_steering_dir $AUGSPEC \
      --alpha_list $4 --top_k_list $5 --vuln_k_list ${6:-0} 2>&1 | tee logs/aug022_$1.log
}

duo() {  # usage: duo <mode> <ranking suffix> <alpha_s> <k_s> <alpha_c> <k_c>
  python steering/duosteer_eval.py --method E --cwe $CWE --model $MODEL --max_new_tokens 2048 \
      --input_file $TASKS --output_dir $OUT/$1/$CWE \
      --safety_causal_results ${SCAUSAL}$2.json --safety_sv_dir $AUGSPEC \
      --correct_causal_results ${CCAUSAL}$2.json \
      --correct_sv_dir_raw $CSV --correct_sv_dir_orth $CSV \
      --alpha1_list $3 --safety_topk_list $4 --alpha2_list $5 --correct_topk_list $6 \
      --steer_positions all --skip_baseline --skip_safety_only 2>&1 | tee logs/aug022_$1.log
}

for s in $STEPS; do
  echo "===== step $s ====="
  case $s in
    1) cmd aug022_cmd_A      ${SCAUSAL}_top256.json all      5 16 ;;
    2) cmd aug022_cmd        ${SCAUSAL}.json        all      3 "4 64" ;;
    3) cmd aug022_cmd_resp   ${SCAUSAL}.json        response 5 16 ;;
    4) cmd aug022_union_resp ${SCAUSAL}.json        response 3 64 4 ;;
    5) duo aug022_duo_A _top256 5 16  5 16 ;;
    6) duo aug022_duo_B ""      3 64 -3 16 ;;
    *) echo "unknown step $s" ;;
  esac
done

# Summary check: Traceback counts must be 0; one file per mode, two for aug022_cmd.
echo "===== check ====="
grep -c Traceback logs/aug022_*.log
for m in $MODES; do
  echo "$m: $(ls $OUT/$m/$CWE/*.jsonl 2>/dev/null | xargs -n1 basename 2>/dev/null | tr '\n' ' ')"
done
