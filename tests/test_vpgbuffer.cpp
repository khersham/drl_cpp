#include <iostream>
#include <cassert>
#include <vector>
#include <cmath> // For std::isnan
#include "torch/torch.h"
#include "drl.h" // Include the header for VPGBuffer
#include "mpi.h" // Required for MPI_Init if we are to test get() which uses MPI

// Basic assertion macros for clarity
#define ASSERT_TRUE(condition, message) \
    if (!(condition)) { \
        std::cerr << "Assertion failed: (" #condition ") " << message << std::endl; \
        MPI_Finalize(); /* Clean up MPI if it was initialized */ \
        std::exit(EXIT_FAILURE); \
    }

#define ASSERT_EQUAL(val1, val2, message) \
    if (!((val1) == (val2))) { \
        std::cerr << "Assertion failed: (" #val1 " == " #val2 ") " \
                  << "Values: " << (val1) << " vs " << (val2) << " " << message << std::endl; \
        MPI_Finalize(); /* Clean up MPI if it was initialized */ \
        std::exit(EXIT_FAILURE); \
    }

// Helper to compare tensors using torch::allclose for floating point numbers
bool tensors_allclose(const torch::Tensor& t1, const torch::Tensor& t2, double rtol=1e-05, double atol=1e-08) {
    if (t1.defined() != t2.defined()) return false;
    if (!t1.defined() && !t2.defined()) return true;
    if (t1.sizes() != t2.sizes()) {
        std::cerr << "tensors_allclose: size mismatch. t1: " << t1.sizes() << ", t2: " << t2.sizes() << std::endl;
        return false;
    }
    return torch::allclose(t1, t2, rtol, atol);
}

// Helper to check if all elements in a tensor are NaN
bool all_elements_nan(const torch::Tensor& t) {
    if (!t.defined() || t.numel() == 0) return false; // Or true if numel is 0, depending on convention
    return torch::all(torch::isnan(t)).item<bool>();
}


void test_vpgbuffer_store_capacity() {
    std::cout << "Running test_vpgbuffer_store_capacity..." << std::endl;
    const int64_t obs_dim = 3;
    const int64_t act_dim = 1;
    const int64_t max_size = 5;
    const float gamma = 0.99f;
    const float lam = 0.97f;

    drl::VPGBuffer buffer(obs_dim, act_dim, max_size, gamma, lam);

    torch::Tensor obs = torch::ones({obs_dim});
    torch::Tensor act = torch::ones({act_dim});
    torch::Tensor rew = torch::tensor(1.0f);
    torch::Tensor val = torch::tensor(0.5f);
    torch::Tensor logp = torch::tensor(-0.1f);

    for (int64_t i = 0; i < max_size; ++i) {
        buffer.store(obs, act, rew, val, logp);
    }
    buffer.store(obs, act, rew, val, logp);
    std::cout << "test_vpgbuffer_store_capacity passed (visual check for 'Buffer is full' needed)." << std::endl;
}

void test_vpgbuffer_discount() {
    std::cout << "Running test_vpgbuffer_discount()..." << std::endl;
    drl::VPGBuffer buffer(1, 1, 1, 0.99f, 0.97f);

    torch::Tensor xarray1 = torch::tensor({1.0f, 1.0f, 1.0f, 1.0f}, torch::kFloat32);
    float gamma1 = 0.9f;
    torch::Tensor expected1 = torch::tensor({3.439f, 2.71f, 1.9f, 1.0f}, torch::kFloat32);
    torch::Tensor result1 = buffer.discount(xarray1, gamma1);
    ASSERT_TRUE(tensors_allclose(result1, expected1), "Discount test case 1 failed.");

    torch::Tensor xarray2 = torch::tensor({1.0f, 2.0f, 3.0f}, torch::kFloat32);
    float gamma2 = 1.0f;
    torch::Tensor expected2 = torch::tensor({6.0f, 5.0f, 3.0f}, torch::kFloat32);
    torch::Tensor result2 = buffer.discount(xarray2, gamma2);
    ASSERT_TRUE(tensors_allclose(result2, expected2), "Discount test case 2 failed.");

    torch::Tensor xarray3 = torch::zeros({0}, torch::kFloat32);
    float gamma3 = 0.9f;
    torch::Tensor expected3 = torch::zeros({0}, torch::kFloat32);
    torch::Tensor result3 = buffer.discount(xarray3, gamma3);
    ASSERT_TRUE(tensors_allclose(result3, expected3), "Discount test case 3 (empty tensor) failed.");

    torch::Tensor xarray4 = torch::tensor({5.0f}, torch::kFloat32);
    float gamma4 = 0.9f;
    torch::Tensor expected4 = torch::tensor({5.0f}, torch::kFloat32);
    torch::Tensor result4 = buffer.discount(xarray4, gamma4);
    ASSERT_TRUE(tensors_allclose(result4, expected4), "Discount test case 4 (single element) failed.");

    std::cout << "test_vpgbuffer_discount() passed." << std::endl;
}

void test_vpgbuffer_finish_path_and_get() {
    std::cout << "Running test_vpgbuffer_finish_path_and_get()..." << std::endl;
    const int64_t obs_dim = 2;
    const int64_t act_dim = 1;
    const int64_t max_size = 3;
    const float gamma = 0.9f;
    const float lam = 0.5f;

    drl::VPGBuffer buffer(obs_dim, act_dim, max_size, gamma, lam);

    buffer.store(torch::tensor({1.0f, 1.0f}), torch::tensor({0.1f}), torch::tensor(1.0f), torch::tensor(0.5f), torch::tensor(-0.1f));
    buffer.store(torch::tensor({2.0f, 2.0f}), torch::tensor({0.2f}), torch::tensor(1.0f), torch::tensor(0.6f), torch::tensor(-0.2f));
    buffer.store(torch::tensor({3.0f, 3.0f}), torch::tensor({0.3f}), torch::tensor(1.0f), torch::tensor(0.7f), torch::tensor(-0.3f));

    torch::Tensor last_val = torch::tensor(0.8f);
    buffer.finish_path(last_val);

    torch::Tensor expected_adv_before_norm = torch::tensor({1.71005f, 1.489f, 1.02f});
    torch::Tensor expected_ret = torch::tensor({3.2932f, 2.548f, 1.72f});

    auto [obs_buf, act_buf, ret_buf, adv_buf, logp_buf] = buffer.get();

    ASSERT_TRUE(tensors_allclose(ret_buf, expected_ret, 1e-4, 1e-4), "Return calculation in finish_path is incorrect.");

    float mean_adv = expected_adv_before_norm.mean().item<float>();
    // Calculate population standard deviation (divide by N) as in mpi_stat
    float std_adv = torch::sqrt(((expected_adv_before_norm - mean_adv).pow(2)).mean()).item<float>();

    torch::Tensor expected_adv_normalized = (expected_adv_before_norm - mean_adv);
    if (std_adv > 1e-8) {
         expected_adv_normalized = expected_adv_normalized / std_adv;
    } else {
        // If std_dev is ~0, normalized advantages should also be ~0 (or NaN if numerator also 0 after float precision)
        // Given numerator (X - mean(X)) will be 0 if all X are same, expect 0/0 = NaN
        // If std_dev is truly 0, then all elements of expected_adv_before_norm are mean_adv.
        // So (expected_adv_before_norm - mean_adv) is a zero tensor.
        // A zero tensor divided by zero std_dev results in NaN.
        expected_adv_normalized = torch::full_like(expected_adv_before_norm, NAN);
    }

    if (all_elements_nan(expected_adv_normalized)) {
         ASSERT_TRUE(all_elements_nan(adv_buf), "Advantage normalization for zero std case (expected NaN) failed.");
    } else {
        ASSERT_TRUE(tensors_allclose(adv_buf, expected_adv_normalized, 1e-4, 1e-4), "Advantage calculation/normalization is incorrect.");
    }

    ASSERT_EQUAL(buffer.ptr, (int64_t)0, "ptr not reset by get()");
    ASSERT_EQUAL(buffer.ptr_start_idx, (int64_t)0, "ptr_start_idx not reset by get()");

    std::cout << "test_vpgbuffer_finish_path_and_get() passed." << std::endl;
}

void test_vpgbuffer_get_specifics_zero_std_adv() {
    std::cout << "Running test_vpgbuffer_get_specifics_zero_std_adv()..." << std::endl;
    const int64_t obs_dim = 1;
    const int64_t act_dim = 1;
    const int64_t max_size = 3;
    const float gamma = 0.9f;
    const float lam = 0.0f; // Set lam = 0 so A_t = delta_t

    drl::VPGBuffer buffer(obs_dim, act_dim, max_size, gamma, lam);

    // Store data that will lead to identical delta values
    // delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
    // Let r_t = 1, V(s_t) = 0.5 for all t. Let V(s_{t+1}) also be 0.5.
    // delta_t = 1 + gamma * 0.5 - 0.5
    // If gamma = 1, delta_t = 1. If gamma = 0.9, delta_t = 1 + 0.45 - 0.5 = 0.95
    torch::Tensor obs = torch::tensor({1.0f});
    torch::Tensor act = torch::tensor({0.1f});
    torch::Tensor rew = torch::tensor(1.0f);      // r_t = 1
    torch::Tensor val = torch::tensor(0.5f);      // V(s_t) = 0.5
    torch::Tensor logp = torch::tensor(-0.1f);

    for (int64_t i = 0; i < max_size; ++i) {
        buffer.store(obs, act, rew, val, logp);
    }

    // last_val for finish_path will be V(s_T)
    // For delta_{T-1} = r_{T-1} + gamma * V(s_T) - V(s_{T-1})
    // We need V(s_T) such that delta_{T-1} is same as other deltas.
    // Let V(s_T) = 0.5f as well.
    torch::Tensor last_val = torch::tensor(0.5f);
    buffer.finish_path(last_val);

    // With lam = 0, adv_buf before normalization should be equal to deltas.
    // delta_t = 1.0f + 0.9f * 0.5f - 0.5f = 1.0f + 0.45f - 0.5f = 0.95f
    // So, adv_buf (before norm) should be [0.95, 0.95, 0.95]
    // Mean is 0.95, std is 0.
    // Normalization: (0.95 - 0.95) / 0 = 0 / 0, which should be NaN.

    auto [obs_b, act_b, ret_b, adv_b, logp_b] = buffer.get();

    // std::cout << "Advantage buffer from get (zero std case): " << adv_b << std::endl;
    ASSERT_TRUE(all_elements_nan(adv_b),
        "Advantage buffer should be all NaNs when pre-normalization std dev is zero.");
    std::cout << "Zero standard deviation advantage normalization test passed (produced NaNs as expected)." << std::endl;

    std::cout << "test_vpgbuffer_get_specifics_zero_std_adv() passed." << std::endl;
}


int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    test_vpgbuffer_store_capacity();
    std::cout << "----" << std::endl;
    test_vpgbuffer_discount();
    std::cout << "----" << std::endl;
    test_vpgbuffer_finish_path_and_get();
    std::cout << "----" << std::endl;
    test_vpgbuffer_get_specifics_zero_std_adv();

    std::cout << "All VPGBuffer tests completed." << std::endl;

    MPI_Finalize();
    return 0;
}
