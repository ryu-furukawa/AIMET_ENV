set -e

function main() {

    while [[ $# -gt 0 ]]; do
        key="$1"
        case $key in
            -a|--sentence_file)
                sentence_file="$2"
                shift 2
                ;;
            -b|--board_seiral_num)
                board_seiral_num="$2"
                shift 2
                ;;
            -c|--board_bin_dir)
                board_bin_dir="$2"
                shift 2
                ;;
            -d|--board_lib_dir)
                board_lib_dir="$2"
                shift 2
                ;;
            -e|--model_config_json)
                model_config_json="$2"
                shift 2
                ;;
            -f|--result_dir)
                result_dir="$2"
                shift 2
                ;;
            *)
                echo "[ERR] Unknown option: $1"
                exit 1
                ;;
        esac
    done


    if [ -n "$(tail -c 1 "$sentence_file")" ]; then
        echo >> "$sentence_file"
    fi

    count=0
    prompt_file=prompt.txt

    while read -r LINE; do

        
        echo "Execution #$count"
        echo "Processing line: $LINE"

        printf "%s" "$LINE" > "$prompt_file"

        mkdir -p $result_dir

        adb  push $prompt_file $board_bin_dir
        
        adb shell "cd $board_bin_dir && \
                sh -x jmteb_embedding_android.sh $board_bin_dir $board_lib_dir $(basename $model_config_json)" </dev/null

        adb  pull $board_bin_dir/output.raw $result_dir/output_$count.raw
        
        let count=count+1

    done < $sentence_file

    rm $prompt_file

}

# The sentence "Could not chdir to home directory /root: No such file or directory" is not output.
exec 2> >(grep -v "Could not chdir to home directory /root: No such file or directory" >&2)

main "$@"
