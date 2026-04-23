from genie_model_fix import main

model_path = "/root/AIMET_ruri/ruri-onnx/model.onnx"
out_path = "/root/AIMET_ruri/ruri-onnx/model_dummy_input.onnx"

if __name__ == "__main__":
    main(model_path, out_path)
