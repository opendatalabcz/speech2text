#!/bin/bash

num_re='^[0-9]+$'
shared_dir="/opt/shared/"
datasets_dir="${shared_dir}/datasets/"

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
        echo "Specify either file id or -r followed by number of randomly generated ids."
        exit 2
fi

if [ "$1" == "-r" ]; then
	if [[ "$2" =~ $num_re ]]; then
		id_list=`tail -n +2 "${datasets_dir}/cpm_cut/test.csv" | sort -R | head -n "$2" | cut -d',' -f1 | rev | cut -d'/' -f1 | rev | cut -d'.' -f1`
		while read -r line
		do
			./inference.sh "$line"
		done <<< "$id_list"
		exit 0
	fi
	echo "After the -r specifier, specify also number of randomly generated inferences. I.e.: ./inference -r 5"
	exit 2
fi

audio_id="$1"

if [ ! -f "${datasets_dir}/cpm_cut/${audio_id}.wav" ]; then
	echo "File ID ${audio_id} does not exist."
	exit 2
fi

echo -n "src: "
cat "${datasets_dir}/cpm_cut/${1}.txt"
echo ""
echo -n "inf: "

best_wer_model=`ls "${shared_dir}/models" | grep pbmm | sort -V | head -n 1`

deepspeech --model "${shared_dir}/models/${best_wer_model}" \
           --lm "${shared_dir}/lm.binary" \
           --trie "${shared_dir}/trie" \
           --audio "${datasets_dir}/cpm_cut/${audio_id}.wav" \
           2> /dev/null

