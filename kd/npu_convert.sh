#!/usr/bin/env bash
# ONNX -> QNN context binary for the OnePlus 15 NPU (QAIRT 2.47, Hexagon v81, soc_model 87).
# Same settings as the phone's SegFormer-B0 (hsv2-lite-app/models/segformer-b0-qairt2.47): v81, O3, VTCM 8 MB.
# Writes ONLY inside $W (its own work folder); uses the shared QAIRT SDK, venvs and NDK read-only.
#
# Usage (on nunnari):  [W=<work folder>] [CALIB=<input_list.txt>] npu_convert.sh <fp16|w8a16>
#   expects  $W/onnx/$NAME.onnx  and (for w8a16) $CALIB (default $W/calib_raw/input_list.txt)
#   produces $W/dist/${NAME}_<mode>.bin  (+ logs in $W/logs)
set -euo pipefail
MODE=$1
NAME=segformer_b1_31  # graph name; same I/O contract for every 31-class student
R=/media/nunnari/projects/qairt
W=${W:-$R/work_247/$NAME}
CALIB=${CALIB:-$W/calib_raw/input_list.txt}
SDK=$R/2.47.0.260601
export TMPDIR=$W/tmp
mkdir -p $W/build/$MODE/model $W/build/$MODE/lib $W/ctx $W/dist $W/logs $TMPDIR

set +eu  # the venv / SDK setup scripts don't work under strict mode
source $R/export_venv_247/bin/activate
source $SDK/bin/envsetup.sh >/dev/null 2>&1
set -eu
export PATH="$R/android-ndk-r26c:$PATH"
command -v qnn-onnx-converter >/dev/null || { echo "QAIRT not on PATH"; exit 1; }

echo "== [1/3] convert ($MODE)"
CMD=(qnn-onnx-converter --input_network $W/onnx/$NAME.onnx --input_dim image 1,512,512,3
     --preserve_io layout image class_logits --output_path $W/build/$MODE/model/$NAME.cpp)
case $MODE in
  fp16)  CMD+=(--float_bitwidth 16) ;;
  w8a16) CMD+=(--input_list $CALIB --act_bitwidth 16 --weights_bitwidth 8
               --bias_bitwidth 32 --use_per_channel_quantization --algorithms cle) ;;
  *) echo "mode must be fp16 or w8a16"; exit 1 ;;
esac
echo "${CMD[*]}" > $W/logs/convert_$MODE.cmd
"${CMD[@]}" > $W/logs/convert_$MODE.log 2>&1 || { tail -30 $W/logs/convert_$MODE.log; exit 1; }
tail -3 $W/logs/convert_$MODE.log

echo "== [2/3] model library (aarch64-android + x86_64-linux-clang)"
qnn-model-lib-generator -c $W/build/$MODE/model/$NAME.cpp -b $W/build/$MODE/model/$NAME.bin \
  -o $W/build/$MODE/lib -l $NAME -t "aarch64-android x86_64-linux-clang" > $W/logs/lib_$MODE.log 2>&1 \
  || { tail -30 $W/logs/lib_$MODE.log; exit 1; }

echo "== [3/3] context binary (v81)"
RELAX=""; [ "$MODE" = fp16 ] && RELAX=', "fp16_relaxed_precision": 1'
cat > $W/ctx/htp_$MODE.json <<EOF
{ "graphs": [ { "graph_names": ["$NAME"], "vtcm_mb": 8, "O": 3 $RELAX } ],
  "devices": [ { "soc_model": 87, "dsp_arch": "v81" } ] }
EOF
cat > $W/ctx/cfg_$MODE.json <<EOF
{"backend_extensions":{"shared_library_path":"$SDK/lib/x86_64-linux-clang/libQnnHtpNetRunExtensions.so","config_file_path":"$W/ctx/htp_$MODE.json"}}
EOF
qnn-context-binary-generator --backend $SDK/lib/x86_64-linux-clang/libQnnHtp.so \
  --model $W/build/$MODE/lib/x86_64-linux-clang/lib$NAME.so --binary_file ${NAME}_$MODE \
  --output_dir $W/dist --config_file $W/ctx/cfg_$MODE.json > $W/logs/ctx_$MODE.log 2>&1 \
  || { tail -30 $W/logs/ctx_$MODE.log; exit 1; }
ls -l $W/dist/${NAME}_$MODE.bin
sha256sum $W/dist/${NAME}_$MODE.bin
