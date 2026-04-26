// FFI shim around `at::convolution_backward`.
//
// Pattern matches torch-sys's auto-generated wrappers:
// - Inputs are `torch::Tensor*` (passed as `void*`)
// - Output is a `tensor*` (i.e. `torch::Tensor**`) whose three slots are
//   filled with `new torch::Tensor(...)` for masks set to true; nullptr otherwise.
// - Errors set the global error string used by `read_and_clean_error` in tch
//   so the Rust side can surface them via the `unsafe_torch_err` macro.

#include <torch/torch.h>
#include <ATen/ATen.h>
#include <array>
#include <cstring>
#include <exception>

extern "C" thread_local char *torch_last_err;

extern "C" void burn_tch_convolution_backward(
    void **out__,
    void *grad_output,
    void *input,
    void *weight,
    int64_t const *bias_sizes_data, size_t bias_sizes_len, int has_bias_sizes,
    int64_t const *stride_data, size_t stride_len,
    int64_t const *padding_data, size_t padding_len,
    int64_t const *dilation_data, size_t dilation_len,
    int transposed,
    int64_t const *output_padding_data, size_t output_padding_len,
    int64_t groups,
    int input_mask, int weight_mask, int bias_mask) {
    out__[0] = nullptr;
    out__[1] = nullptr;
    out__[2] = nullptr;
    try {
        const torch::Tensor &go = *static_cast<const torch::Tensor *>(grad_output);
        const torch::Tensor &in = *static_cast<const torch::Tensor *>(input);
        const torch::Tensor &w = *static_cast<const torch::Tensor *>(weight);

        c10::OptionalArrayRef<int64_t> bias_sizes_opt;
        if (has_bias_sizes) {
            bias_sizes_opt = c10::ArrayRef<int64_t>(bias_sizes_data, bias_sizes_len);
        }

        std::array<bool, 3> output_mask{
            input_mask != 0, weight_mask != 0, bias_mask != 0};

        auto result = at::convolution_backward(
            go, in, w,
            bias_sizes_opt,
            c10::IntArrayRef(stride_data, stride_len),
            c10::IntArrayRef(padding_data, padding_len),
            c10::IntArrayRef(dilation_data, dilation_len),
            transposed != 0,
            c10::IntArrayRef(output_padding_data, output_padding_len),
            groups,
            output_mask);

        if (input_mask && std::get<0>(result).defined()) {
            out__[0] = new torch::Tensor(std::get<0>(result));
        }
        if (weight_mask && std::get<1>(result).defined()) {
            out__[1] = new torch::Tensor(std::get<1>(result));
        }
        if (bias_mask && std::get<2>(result).defined()) {
            out__[2] = new torch::Tensor(std::get<2>(result));
        }
    } catch (const std::exception &e) {
        torch_last_err = strdup(e.what());
    }
}
