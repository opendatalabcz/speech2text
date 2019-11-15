#!/bin/bash

num_re='^[0-9]+$'
shared_dir="/opt/shared/"
datasets_dir="${shared}datasets/"

if [ "$#" -ls 2 ]; then
        echo "Specify either file id or -r followed by number of randomly generated ids."
        exit 2
fi

if [ "$1" == "-r" ]; then
	if [[ "$2" =~ $num_re ]]; then
		id_list=`tail -n +2 "${datasets_dir}cpm_cut/test.csv" | sort -R | head | cut -d',' -f1 | cut -d'/' -f5 | cut -d'.' -f1`
		while read -r line
		do
			./inference.sh "$line"
		done <<< "$id_list"
	fi
	echo "After the -r specifier, specify also number of randomly generated inferences. I.e.: ./inference -r 5"
	exit 2
fi

audio_id="$1"

if [ ! -f "${datasets_dir}/cpm_cut/${audio_id}" ]; then
	echo "File ID ${audio_id} does not exist."
	exit 2
fi

echo -n "src: "
cat "${datasets_dir}cpm_cut/${1}.txt"
echo ""
echo -n "inf: "

deepspeech --model "${shared_dir}output_graph.pbmm" \
           --lm "${shared_dir}lm.binary" \
           --trie "${shared_dir}trie" \
           --audio "${datasets_dir}cpm_cut/${audio_id}.wav" \
           --alphabet "${datasets_dir}alphabet_cz.txt" \
           2> /dev/null

