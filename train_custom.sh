printf -v curr_date '%(%Y%m%d-%H%M%S)T' -1
log_fn="${curr_date}.log"

model_dir="../shared/models/"
shared_dir="/opt/shared/"
datasets_dir="${shared_dir}/datasets/"

echo "Logging to: ${log_fn}"

mkdir -p "../ds_outputs/export" "../ds_outputs/checkpoints" "../ds_outputs/summary" "../ds_outputs/logs"

######-PARAMS-######
train_files="${datasets_dir}/cpm_cut/train.csv"
test_files="${datasets_dir}/cpm_cut/test.csv"
dev_files="${datasets_dir}/cpm_cut/dev.csv"
alphabet_config_path="${shared_dir}/alphabet_cz.txt"
export_dir="../ds_outputs/export"
checkpoint_dir="../ds_outputs/checkpoints"
summary_dir="../ds_outputs/summary"
log_dir="../ds_outputs/logs"
lm_binary_path="${shared_dir}/lm.binary"
lm_trie_path="${shared_dir}/trie"
####################
train_batch_size=24
dev_batch_size=44
test_batch_size=44
n_hidden=2048
learning_rate=0.0001
dropout_rate=0.2
epochs=25
early_stop="true"
lm_alpha=0.75
lm_beta=1.85
###################


./DeepSpeech.py \
 --train_files "$train_files" \
 --test_files "$test_files" \
 --dev_files "$dev_files" \
 --alphabet_config_path "$alphabet_config_path" \
 --export_dir "$export_dir" \
 --checkpoint_dir "$checkpoint_dir" \
 --summary_dir "$summary_dir" \
 --log_dir "$log_dir" \
 --lm_binary_path "$lm_binary_path" \
 --lm_trie_path "$lm_trie_path" \
 --train_batch_size "$train_batch_size" \
 --dev_batch_size "$dev_batch_size" \
 --test_batch_size "$test_batch_size" \
 --n_hidden "$n_hidden" \
 --learning_rate "$learning_rate" \
 --dropout_rate "$dropout_rate" \
 --epochs "$epochs" \
 --early_stop "$early_stop" \
 --lm_alpha "$lm_alpha" \
 --lm_beta "$lm_beta" \
 --use_allow_growth \
 --automatic_mixed_precision > "${log_fn}" 2>&1

test_WER=`grep "Test on" "${log_fn}" | cut -d':' -f2 | cut -d',' -f1 | tr -d ' '`
test_CER=`grep "Test on" "${log_fn}" | cut -d':' -f3 | cut -d',' -f1 | tr -d ' '`
test_loss=`grep "Test on" "${log_fn}" | cut -d':' -f4 | cut -d',' -f1 | tr -d ' '`

model_info_fn="${model_dir}${test_WER}_${test_CER}_${test_loss}"

./native_client_bin/convert_graphdef_memmapped_format --in_graph="../ds_outputs/export/output_graph.pb" \
						      --out_graph="${model_info_fn}.pbmm" >> "${log_fn}" 2>&1

if [ -f "${model_info_fn}.pbmm" ]; then
	rm "../ds_outputs/export/output_graph.pb"
fi

echo "Batch sizes (tr/de/te):${train_batch_size}/${dev_batch_size}/${test_batch_size}" >> "${model_info_fn}.txt"
echo "N-hidden:${n_hidden}" >> "${model_info_fn}.txt"
echo "Learning rate:${learning_rate}" >> "${model_info_fn}.txt"
echo "Dropout:${dropout_rate}" >> "${model_info_fn}.txt"
echo "Epochs:${epochs}" >> "${model_info_fn}.txt"
echo "Early stop:${early_stop}" >> "${model_info_fn}.txt"
echo "lm_alpha:${lm_alpha}" >> "${model_info_fn}.txt"
echo "lm_beta:${lm_beta}" >> "${model_info_fn}.txt"
