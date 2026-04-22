set -e

export PATH=$1:$PATH
export VENDOR_LIB=$2
export LD_LIBRARY_PATH=$VENDOR_LIB:$LD_LIBRARY_PATH
export ADSP_LIBRARY_PATH="/vendor/lib/rfsa/adsp;$VENDOR_LIB;"


./genie-t2e-run \
    -c "${3}" \
    --prompt_file prompt.txt

                    