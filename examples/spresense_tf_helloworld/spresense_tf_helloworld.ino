#include <TensorFlowLite.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace {
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
int inference_count = 0;

constexpr int kTensorArenaSize = 2000;
uint8_t tensor_arena[kTensorArenaSize];
}  // namespace


const float kXrange = 2.f * 3.14159265359f;
const int kInferencesPerCycle = 20;

extern const unsigned char g_model[];
extern const int g_model_len;

void setup() {
  
  Serial.begin(115200);
  tflite::InitializeTarget();

  // Set up logging. 
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Map the model into a usable data structure..
  model = tflite::GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model provided is schema version " 
                  + String(model->version()) + " not equal "
                  + "to supported version "
                  + String(TFLITE_SCHEMA_VERSION));
    return;
  } else {
    Serial.println("Model version: " + String(model->version()));
  }

  // This pulls in all the operation implementations we need.
  static tflite::AllOpsResolver resolver;

  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    Serial.println("AllocateTensors() failed");
    return;
  }

  // Obtain pointers to the model's input and output tensors.
  input = interpreter->input(0);
  output = interpreter->output(0);

  // Keep track of how many inferences we have performed.
  inference_count = 0;
}

void loop() {
  // Calculate an x value to feed into the model.
  float position_x = static_cast<float>(inference_count) /
                   static_cast<float>(kInferencesPerCycle);
  float x = position_x * kXrange;

  // Quantize the input from floating-point to integer
  int8_t x_quantized = x / input->params.scale + input->params.zero_point;

  // Place the quantized input in the model's input tensor
  input->data.int8[0] = x_quantized;

  // Run inference, and report any error
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    Serial.println("Invoke failed on x: " + String(x));
    return;
  }

  // Obtain the quantized output from model's output tensor
  int8_t y_quantized = output->data.int8[0];
  
  // Dequantize the output from integer to floating-point
  float y = (y_quantized - output->params.zero_point) * output->params.scale;
  Serial.println("y_value: " + String(y));

  ++inference_count;
  if (inference_count >= kInferencesPerCycle) inference_count = 0;
}
