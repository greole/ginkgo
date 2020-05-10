/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2020, the Ginkgo authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#include "core/matrix/csr_kernels.hpp"


#include <algorithm>
#include <iterator>
#include <numeric>
#include <utility>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/ell.hpp>
#include <ginkgo/core/matrix/hybrid.hpp>
#include <ginkgo/core/matrix/sellp.hpp>


#include "core/base/allocator.hpp"
#include "core/base/iterator_factory.hpp"
#include "core/components/prefix_sum.hpp"
#include "core/matrix/csr_builder.hpp"
#include "reference/components/format_conversion.hpp"


namespace gko {
namespace kernels {
namespace reference {
/**
 * @brief The Compressed sparse row matrix format namespace.
 * @ref Csr
 * @ingroup csr
 */
namespace csr {


template <typename ValueType, typename IndexType>
void spmv(std::shared_ptr<const ReferenceExecutor> exec,
          const matrix::Csr<ValueType, IndexType> *a,
          const matrix::Dense<ValueType> *b, matrix::Dense<ValueType> *c)
{
    auto row_ptrs = a->get_const_row_ptrs();
    auto col_idxs = a->get_const_col_idxs();
    auto vals = a->get_const_values();

    for (size_type row = 0; row < a->get_size()[0]; ++row) {
        for (size_type j = 0; j < c->get_size()[1]; ++j) {
            c->at(row, j) = zero<ValueType>();
        }
        for (size_type k = row_ptrs[row];
             k < static_cast<size_type>(row_ptrs[row + 1]); ++k) {
            auto val = vals[k];
            auto col = col_idxs[k];
            for (size_type j = 0; j < c->get_size()[1]; ++j) {
                c->at(row, j) += val * b->at(col, j);
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_CSR_SPMV_KERNEL);


template <typename ValueType, typename IndexType>
void advanced_spmv(std::shared_ptr<const ReferenceExecutor> exec,
                   const matrix::Dense<ValueType> *alpha,
                   const matrix::Csr<ValueType, IndexType> *a,
                   const matrix::Dense<ValueType> *b,
                   const matrix::Dense<ValueType> *beta,
                   matrix::Dense<ValueType> *c)
{
    auto row_ptrs = a->get_const_row_ptrs();
    auto col_idxs = a->get_const_col_idxs();
    auto vals = a->get_const_values();
    auto valpha = alpha->at(0, 0);
    auto vbeta = beta->at(0, 0);

    for (size_type row = 0; row < a->get_size()[0]; ++row) {
        for (size_type j = 0; j < c->get_size()[1]; ++j) {
            c->at(row, j) *= vbeta;
        }
        for (size_type k = row_ptrs[row];
             k < static_cast<size_type>(row_ptrs[row + 1]); ++k) {
            auto val = vals[k];
            auto col = col_idxs[k];
            for (size_type j = 0; j < c->get_size()[1]; ++j) {
                c->at(row, j) += valpha * val * b->at(col, j);
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_ADVANCED_SPMV_KERNEL);


template <typename ValueType, typename IndexType>
void spgemm_insert_row(unordered_set<IndexType> &cols,
                       const matrix::Csr<ValueType, IndexType> *c,
                       size_type row)
{
    auto row_ptrs = c->get_const_row_ptrs();
    auto col_idxs = c->get_const_col_idxs();
    cols.insert(col_idxs + row_ptrs[row], col_idxs + row_ptrs[row + 1]);
}


template <typename ValueType, typename IndexType>
void spgemm_insert_row2(unordered_set<IndexType> &cols,
                        const matrix::Csr<ValueType, IndexType> *a,
                        const matrix::Csr<ValueType, IndexType> *b,
                        size_type row)
{
    auto a_row_ptrs = a->get_const_row_ptrs();
    auto a_col_idxs = a->get_const_col_idxs();
    auto b_row_ptrs = b->get_const_row_ptrs();
    auto b_col_idxs = b->get_const_col_idxs();
    for (size_type a_nz = a_row_ptrs[row];
         a_nz < size_type(a_row_ptrs[row + 1]); ++a_nz) {
        auto a_col = a_col_idxs[a_nz];
        auto b_row = a_col;
        cols.insert(b_col_idxs + b_row_ptrs[b_row],
                    b_col_idxs + b_row_ptrs[b_row + 1]);
    }
}


template <typename ValueType, typename IndexType>
void spgemm_accumulate_row(map<IndexType, ValueType> &cols,
                           const matrix::Csr<ValueType, IndexType> *c,
                           ValueType scale, size_type row)
{
    auto row_ptrs = c->get_const_row_ptrs();
    auto col_idxs = c->get_const_col_idxs();
    auto vals = c->get_const_values();
    for (size_type c_nz = row_ptrs[row]; c_nz < size_type(row_ptrs[row + 1]);
         ++c_nz) {
        auto c_col = col_idxs[c_nz];
        auto c_val = vals[c_nz];
        cols[c_col] += scale * c_val;
    }
}


template <typename ValueType, typename IndexType>
void spgemm_accumulate_row2(map<IndexType, ValueType> &cols,
                            const matrix::Csr<ValueType, IndexType> *a,
                            const matrix::Csr<ValueType, IndexType> *b,
                            ValueType scale, size_type row)
{
    auto a_row_ptrs = a->get_const_row_ptrs();
    auto a_col_idxs = a->get_const_col_idxs();
    auto a_vals = a->get_const_values();
    auto b_row_ptrs = b->get_const_row_ptrs();
    auto b_col_idxs = b->get_const_col_idxs();
    auto b_vals = b->get_const_values();
    for (size_type a_nz = a_row_ptrs[row];
         a_nz < size_type(a_row_ptrs[row + 1]); ++a_nz) {
        auto a_col = a_col_idxs[a_nz];
        auto a_val = a_vals[a_nz];
        auto b_row = a_col;
        for (size_type b_nz = b_row_ptrs[b_row];
             b_nz < size_type(b_row_ptrs[b_row + 1]); ++b_nz) {
            auto b_col = b_col_idxs[b_nz];
            auto b_val = b_vals[b_nz];
            cols[b_col] += scale * a_val * b_val;
        }
    }
}


template <typename ValueType, typename IndexType>
void spgemm(std::shared_ptr<const ReferenceExecutor> exec,
            const matrix::Csr<ValueType, IndexType> *a,
            const matrix::Csr<ValueType, IndexType> *b,
            matrix::Csr<ValueType, IndexType> *c)
{
    auto num_rows = a->get_size()[0];

    // first sweep: count nnz for each row
    auto c_row_ptrs = c->get_row_ptrs();

    unordered_set<IndexType> local_col_idxs(exec);
    for (size_type a_row = 0; a_row < num_rows; ++a_row) {
        local_col_idxs.clear();
        spgemm_insert_row2(local_col_idxs, a, b, a_row);
        c_row_ptrs[a_row] = local_col_idxs.size();
    }

    // build row pointers
    components::prefix_sum(exec, c_row_ptrs, num_rows + 1);

    // second sweep: accumulate non-zeros
    auto new_nnz = c_row_ptrs[num_rows];
    matrix::CsrBuilder<ValueType, IndexType> c_builder{c};
    auto &c_col_idxs_array = c_builder.get_col_idx_array();
    auto &c_vals_array = c_builder.get_value_array();
    c_col_idxs_array.resize_and_reset(new_nnz);
    c_vals_array.resize_and_reset(new_nnz);
    auto c_col_idxs = c_col_idxs_array.get_data();
    auto c_vals = c_vals_array.get_data();

    map<IndexType, ValueType> local_row_nzs(exec);
    for (size_type a_row = 0; a_row < num_rows; ++a_row) {
        local_row_nzs.clear();
        spgemm_accumulate_row2(local_row_nzs, a, b, one<ValueType>(), a_row);
        // store result
        auto c_nz = c_row_ptrs[a_row];
        for (auto pair : local_row_nzs) {
            c_col_idxs[c_nz] = pair.first;
            c_vals[c_nz] = pair.second;
            ++c_nz;
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_CSR_SPGEMM_KERNEL);


template <typename ValueType, typename IndexType>
void advanced_spgemm(std::shared_ptr<const ReferenceExecutor> exec,
                     const matrix::Dense<ValueType> *alpha,
                     const matrix::Csr<ValueType, IndexType> *a,
                     const matrix::Csr<ValueType, IndexType> *b,
                     const matrix::Dense<ValueType> *beta,
                     const matrix::Csr<ValueType, IndexType> *d,
                     matrix::Csr<ValueType, IndexType> *c)
{
    auto num_rows = a->get_size()[0];
    auto valpha = alpha->at(0, 0);
    auto vbeta = beta->at(0, 0);

    // first sweep: count nnz for each row
    auto c_row_ptrs = c->get_row_ptrs();

    unordered_set<IndexType> local_col_idxs(exec);
    for (size_type a_row = 0; a_row < num_rows; ++a_row) {
        local_col_idxs.clear();
        if (vbeta != zero(vbeta)) {
            spgemm_insert_row(local_col_idxs, d, a_row);
        }
        if (valpha != zero(valpha)) {
            spgemm_insert_row2(local_col_idxs, a, b, a_row);
        }
        c_row_ptrs[a_row] = local_col_idxs.size();
    }

    // build row pointers
    components::prefix_sum(exec, c_row_ptrs, num_rows + 1);

    // second sweep: accumulate non-zeros
    auto new_nnz = c_row_ptrs[num_rows];
    matrix::CsrBuilder<ValueType, IndexType> c_builder{c};
    auto &c_col_idxs_array = c_builder.get_col_idx_array();
    auto &c_vals_array = c_builder.get_value_array();
    c_col_idxs_array.resize_and_reset(new_nnz);
    c_vals_array.resize_and_reset(new_nnz);
    auto c_col_idxs = c_col_idxs_array.get_data();
    auto c_vals = c_vals_array.get_data();

    map<IndexType, ValueType> local_row_nzs(exec);
    for (size_type a_row = 0; a_row < num_rows; ++a_row) {
        local_row_nzs.clear();
        if (vbeta != zero(vbeta)) {
            spgemm_accumulate_row(local_row_nzs, d, vbeta, a_row);
        }
        if (valpha != zero(valpha)) {
            spgemm_accumulate_row2(local_row_nzs, a, b, valpha, a_row);
        }
        // store result
        auto c_nz = c_row_ptrs[a_row];
        for (auto pair : local_row_nzs) {
            c_col_idxs[c_nz] = pair.first;
            c_vals[c_nz] = pair.second;
            ++c_nz;
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_ADVANCED_SPGEMM_KERNEL);


template <typename IndexType>
void convert_row_ptrs_to_idxs(std::shared_ptr<const ReferenceExecutor> exec,
                              const IndexType *ptrs, size_type num_rows,
                              IndexType *idxs)
{
    convert_ptrs_to_idxs(ptrs, num_rows, idxs);
}


template <typename ValueType, typename IndexType>
void convert_to_coo(std::shared_ptr<const ReferenceExecutor> exec,
                    const matrix::Csr<ValueType, IndexType> *source,
                    matrix::Coo<ValueType, IndexType> *result)
{
    auto num_rows = result->get_size()[0];

    auto row_idxs = result->get_row_idxs();
    const auto source_row_ptrs = source->get_const_row_ptrs();

    convert_row_ptrs_to_idxs(exec, source_row_ptrs, num_rows, row_idxs);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_CONVERT_TO_COO_KERNEL);

template <typename ValueType, typename IndexType>
void convert_to_dense(std::shared_ptr<const ReferenceExecutor> exec,
                      const matrix::Csr<ValueType, IndexType> *source,
                      matrix::Dense<ValueType> *result)
{
    auto num_rows = source->get_size()[0];
    auto num_cols = source->get_size()[1];
    auto row_ptrs = source->get_const_row_ptrs();
    auto col_idxs = source->get_const_col_idxs();
    auto vals = source->get_const_values();

    for (size_type row = 0; row < num_rows; ++row) {
        for (size_type col = 0; col < num_cols; ++col) {
            result->at(row, col) = zero<ValueType>();
        }
        for (size_type i = row_ptrs[row];
             i < static_cast<size_type>(row_ptrs[row + 1]); ++i) {
            result->at(row, col_idxs[i]) = vals[i];
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_CONVERT_TO_DENSE_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_sellp(std::shared_ptr<const ReferenceExecutor> exec,
                      const matrix::Csr<ValueType, IndexType> *source,
                      matrix::Sellp<ValueType, IndexType> *result)
{
    auto num_rows = result->get_size()[0];
    auto num_cols = result->get_size()[1];
    auto vals = result->get_values();
    auto col_idxs = result->get_col_idxs();
    auto slice_lengths = result->get_slice_lengths();
    auto slice_sets = result->get_slice_sets();
    auto slice_size = (result->get_slice_size() == 0)
                          ? matrix::default_slice_size
                          : result->get_slice_size();
    auto stride_factor = (result->get_stride_factor() == 0)
                             ? matrix::default_stride_factor
                             : result->get_stride_factor();

    const auto source_row_ptrs = source->get_const_row_ptrs();
    const auto source_col_idxs = source->get_const_col_idxs();
    const auto source_values = source->get_const_values();

    int slice_num = ceildiv(num_rows, slice_size);
    slice_sets[0] = 0;
    for (size_type slice = 0; slice < slice_num; slice++) {
        if (slice > 0) {
            slice_sets[slice] =
                slice_sets[slice - 1] + slice_lengths[slice - 1];
        }
        slice_lengths[slice] = 0;
        for (size_type row = 0; row < slice_size; row++) {
            size_type global_row = slice * slice_size + row;
            if (global_row >= num_rows) {
                break;
            }
            slice_lengths[slice] =
                (slice_lengths[slice] >
                 source_row_ptrs[global_row + 1] - source_row_ptrs[global_row])
                    ? slice_lengths[slice]
                    : source_row_ptrs[global_row + 1] -
                          source_row_ptrs[global_row];
        }
        slice_lengths[slice] =
            stride_factor * ceildiv(slice_lengths[slice], stride_factor);
        for (size_type row = 0; row < slice_size; row++) {
            size_type global_row = slice * slice_size + row;
            if (global_row >= num_rows) {
                break;
            }
            size_type sellp_ind = slice_sets[slice] * slice_size + row;
            for (size_type csr_ind = source_row_ptrs[global_row];
                 csr_ind < source_row_ptrs[global_row + 1]; csr_ind++) {
                vals[sellp_ind] = source_values[csr_ind];
                col_idxs[sellp_ind] = source_col_idxs[csr_ind];
                sellp_ind += slice_size;
            }
            for (size_type i = sellp_ind;
                 i <
                 (slice_sets[slice] + slice_lengths[slice]) * slice_size + row;
                 i += slice_size) {
                col_idxs[i] = 0;
                vals[i] = zero<ValueType>();
            }
        }
    }
    if (slice_num > 0) {
        slice_sets[slice_num] =
            slice_sets[slice_num - 1] + slice_lengths[slice_num - 1];
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_CONVERT_TO_SELLP_KERNEL);


template <typename ValueType, typename IndexType>
void calculate_total_cols(std::shared_ptr<const ReferenceExecutor> exec,
                          const matrix::Csr<ValueType, IndexType> *source,
                          size_type *result, size_type stride_factor,
                          size_type slice_size)
{
    size_type total_cols = 0;
    const auto num_rows = source->get_size()[0];
    const auto slice_num = ceildiv(num_rows, slice_size);

    const auto row_ptrs = source->get_const_row_ptrs();

    for (size_type slice = 0; slice < slice_num; slice++) {
        IndexType max_nnz_per_row_in_this_slice = 0;
        for (size_type row = 0;
             row < slice_size && row + slice * slice_size < num_rows; row++) {
            size_type global_row = slice * slice_size + row;
            max_nnz_per_row_in_this_slice =
                max(row_ptrs[global_row + 1] - row_ptrs[global_row],
                    max_nnz_per_row_in_this_slice);
        }
        total_cols += ceildiv(max_nnz_per_row_in_this_slice, stride_factor) *
                      stride_factor;
    }

    *result = total_cols;
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_CALCULATE_TOTAL_COLS_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_ell(std::shared_ptr<const ReferenceExecutor> exec,
                    const matrix::Csr<ValueType, IndexType> *source,
                    matrix::Ell<ValueType, IndexType> *result)
{
    const auto num_rows = source->get_size()[0];
    const auto num_cols = source->get_size()[1];
    const auto vals = source->get_const_values();
    const auto col_idxs = source->get_const_col_idxs();
    const auto row_ptrs = source->get_const_row_ptrs();

    const auto num_stored_elements_per_row =
        result->get_num_stored_elements_per_row();

    for (size_type row = 0; row < num_rows; row++) {
        for (size_type i = 0; i < num_stored_elements_per_row; i++) {
            result->val_at(row, i) = zero<ValueType>();
            result->col_at(row, i) = 0;
        }
        for (size_type col_idx = 0; col_idx < row_ptrs[row + 1] - row_ptrs[row];
             col_idx++) {
            result->val_at(row, col_idx) = vals[row_ptrs[row] + col_idx];
            result->col_at(row, col_idx) = col_idxs[row_ptrs[row] + col_idx];
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_CONVERT_TO_ELL_KERNEL);


template <typename ValueType, typename IndexType, typename UnaryOperator>
inline void convert_csr_to_csc(size_type num_rows, const IndexType *row_ptrs,
                               const IndexType *col_idxs,
                               const ValueType *csr_vals, IndexType *row_idxs,
                               IndexType *col_ptrs, ValueType *csc_vals,
                               UnaryOperator op)
{
    for (size_type row = 0; row < num_rows; ++row) {
        for (auto i = row_ptrs[row]; i < row_ptrs[row + 1]; ++i) {
            const auto dest_idx = col_ptrs[col_idxs[i]]++;
            row_idxs[dest_idx] = row;
            csc_vals[dest_idx] = op(csr_vals[i]);
        }
    }
}


template <typename ValueType, typename IndexType, typename UnaryOperator>
void transpose_and_transform(std::shared_ptr<const ReferenceExecutor> exec,
                             matrix::Csr<ValueType, IndexType> *trans,
                             const matrix::Csr<ValueType, IndexType> *orig,
                             UnaryOperator op)
{
    auto trans_row_ptrs = trans->get_row_ptrs();
    auto orig_row_ptrs = orig->get_const_row_ptrs();
    auto trans_col_idxs = trans->get_col_idxs();
    auto orig_col_idxs = orig->get_const_col_idxs();
    auto trans_vals = trans->get_values();
    auto orig_vals = orig->get_const_values();

    auto orig_num_cols = orig->get_size()[1];
    auto orig_num_rows = orig->get_size()[0];
    auto orig_nnz = orig_row_ptrs[orig_num_rows];

    trans_row_ptrs[0] = 0;
    convert_idxs_to_ptrs(orig_col_idxs, orig_nnz, trans_row_ptrs + 1,
                         orig_num_cols);

    convert_csr_to_csc(orig_num_rows, orig_row_ptrs, orig_col_idxs, orig_vals,
                       trans_col_idxs, trans_row_ptrs + 1, trans_vals, op);
}


template <typename ValueType, typename IndexType>
void transpose(std::shared_ptr<const ReferenceExecutor> exec,
               const matrix::Csr<ValueType, IndexType> *orig,
               matrix::Csr<ValueType, IndexType> *trans)
{
    transpose_and_transform(exec, trans, orig,
                            [](const ValueType x) { return x; });
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_CSR_TRANSPOSE_KERNEL);


template <typename ValueType, typename IndexType>
void conj_transpose(std::shared_ptr<const ReferenceExecutor> exec,
                    const matrix::Csr<ValueType, IndexType> *orig,
                    matrix::Csr<ValueType, IndexType> *trans)
{
    transpose_and_transform(exec, trans, orig,
                            [](const ValueType x) { return conj(x); });
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_CONJ_TRANSPOSE_KERNEL);


template <typename ValueType, typename IndexType>
void calculate_max_nnz_per_row(std::shared_ptr<const ReferenceExecutor> exec,
                               const matrix::Csr<ValueType, IndexType> *source,
                               size_type *result)
{
    const auto num_rows = source->get_size()[0];
    const auto row_ptrs = source->get_const_row_ptrs();
    IndexType max_nnz = 0;

    for (size_type i = 0; i < num_rows; i++) {
        max_nnz = max(row_ptrs[i + 1] - row_ptrs[i], max_nnz);
    }

    *result = max_nnz;
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_CALCULATE_MAX_NNZ_PER_ROW_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_hybrid(std::shared_ptr<const ReferenceExecutor> exec,
                       const matrix::Csr<ValueType, IndexType> *source,
                       matrix::Hybrid<ValueType, IndexType> *result)
{
    auto num_rows = result->get_size()[0];
    auto num_cols = result->get_size()[1];
    auto strategy = result->get_strategy();
    auto ell_lim = strategy->get_ell_num_stored_elements_per_row();
    auto coo_lim = strategy->get_coo_nnz();
    auto coo_val = result->get_coo_values();
    auto coo_col = result->get_coo_col_idxs();
    auto coo_row = result->get_coo_row_idxs();

    // Initial Hybrid Matrix
    for (size_type i = 0; i < result->get_ell_num_stored_elements_per_row();
         i++) {
        for (size_type j = 0; j < result->get_ell_stride(); j++) {
            result->ell_val_at(j, i) = zero<ValueType>();
            result->ell_col_at(j, i) = 0;
        }
    }
    for (size_type i = 0; i < result->get_coo_num_stored_elements(); i++) {
        coo_val[i] = zero<ValueType>();
        coo_col[i] = 0;
        coo_row[i] = 0;
    }

    const auto csr_row_ptrs = source->get_const_row_ptrs();
    const auto csr_vals = source->get_const_values();
    size_type csr_idx = 0;
    size_type coo_idx = 0;
    for (IndexType row = 0; row < num_rows; row++) {
        size_type ell_idx = 0;
        while (csr_idx < csr_row_ptrs[row + 1]) {
            const auto val = csr_vals[csr_idx];
            if (ell_idx < ell_lim) {
                result->ell_val_at(row, ell_idx) = val;
                result->ell_col_at(row, ell_idx) =
                    source->get_const_col_idxs()[csr_idx];
                ell_idx++;
            } else {
                coo_val[coo_idx] = val;
                coo_col[coo_idx] = source->get_const_col_idxs()[csr_idx];
                coo_row[coo_idx] = row;
                coo_idx++;
            }
            csr_idx++;
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_CONVERT_TO_HYBRID_KERNEL);


template <typename ValueType, typename IndexType>
void row_permute_impl(std::shared_ptr<const ReferenceExecutor> exec,
                      const Array<IndexType> *permutation_indices,
                      const matrix::Csr<ValueType, IndexType> *orig,
                      matrix::Csr<ValueType, IndexType> *row_permuted)
{
    auto perm = permutation_indices->get_const_data();
    auto orig_row_ptrs = orig->get_const_row_ptrs();
    auto orig_col_idxs = orig->get_const_col_idxs();
    auto orig_vals = orig->get_const_values();
    auto rp_row_ptrs = row_permuted->get_row_ptrs();
    auto rp_col_idxs = row_permuted->get_col_idxs();
    auto rp_vals = row_permuted->get_values();
    size_type num_rows = orig->get_size()[0];
    size_type num_nnz = orig->get_num_stored_elements();

    size_type cur_ptr = 0;
    rp_row_ptrs[0] = cur_ptr;
    vector<size_type> orig_num_nnz_per_row(num_rows, 0, exec);
    for (size_type row = 0; row < num_rows; ++row) {
        orig_num_nnz_per_row[row] = orig_row_ptrs[row + 1] - orig_row_ptrs[row];
    }
    for (size_type row = 0; row < num_rows; ++row) {
        rp_row_ptrs[row + 1] =
            rp_row_ptrs[row] + orig_num_nnz_per_row[perm[row]];
    }
    rp_row_ptrs[num_rows] = orig_row_ptrs[num_rows];
    for (size_type row = 0; row < num_rows; ++row) {
        auto new_row = perm[row];
        auto new_k = orig_row_ptrs[new_row];
        for (size_type k = rp_row_ptrs[row];
             k < size_type(rp_row_ptrs[row + 1]); ++k) {
            rp_col_idxs[k] = orig_col_idxs[new_k];
            rp_vals[k] = orig_vals[new_k];
            new_k++;
        }
    }
}


template <typename ValueType, typename IndexType>
void row_permute(std::shared_ptr<const ReferenceExecutor> exec,
                 const Array<IndexType> *permutation_indices,
                 const matrix::Csr<ValueType, IndexType> *orig,
                 matrix::Csr<ValueType, IndexType> *row_permuted)
{
    row_permute_impl(exec, permutation_indices, orig, row_permuted);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_ROW_PERMUTE_KERNEL);


template <typename ValueType, typename IndexType>
void inverse_row_permute(std::shared_ptr<const ReferenceExecutor> exec,
                         const Array<IndexType> *permutation_indices,
                         const matrix::Csr<ValueType, IndexType> *orig,
                         matrix::Csr<ValueType, IndexType> *row_permuted)
{
    auto perm = permutation_indices->get_const_data();
    Array<IndexType> inv_perm(*permutation_indices);
    auto iperm = inv_perm.get_data();
    for (size_type ind = 0; ind < inv_perm.get_num_elems(); ++ind) {
        iperm[perm[ind]] = ind;
    }

    row_permute_impl(exec, &inv_perm, orig, row_permuted);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_INVERSE_ROW_PERMUTE_KERNEL);


template <typename ValueType, typename IndexType>
void column_permute_impl(const Array<IndexType> *permutation_indices,
                         const matrix::Csr<ValueType, IndexType> *orig,
                         matrix::Csr<ValueType, IndexType> *column_permuted)
{
    auto perm = permutation_indices->get_const_data();
    auto orig_row_ptrs = orig->get_const_row_ptrs();
    auto orig_col_idxs = orig->get_const_col_idxs();
    auto orig_vals = orig->get_const_values();
    auto cp_row_ptrs = column_permuted->get_row_ptrs();
    auto cp_col_idxs = column_permuted->get_col_idxs();
    auto cp_vals = column_permuted->get_values();
    auto num_nnz = orig->get_num_stored_elements();
    size_type num_rows = orig->get_size()[0];
    size_type num_cols = orig->get_size()[1];

    for (size_type row = 0; row < num_rows; ++row) {
        cp_row_ptrs[row] = orig_row_ptrs[row];
        for (size_type k = orig_row_ptrs[row];
             k < size_type(orig_row_ptrs[row + 1]); ++k) {
            cp_col_idxs[k] = perm[orig_col_idxs[k]];
            cp_vals[k] = orig_vals[k];
        }
    }
    cp_row_ptrs[num_rows] = orig_row_ptrs[num_rows];
}


template <typename ValueType, typename IndexType>
void column_permute(std::shared_ptr<const ReferenceExecutor> exec,
                    const Array<IndexType> *permutation_indices,
                    const matrix::Csr<ValueType, IndexType> *orig,
                    matrix::Csr<ValueType, IndexType> *column_permuted)
{
    auto perm = permutation_indices->get_const_data();
    Array<IndexType> inv_perm(*permutation_indices);
    auto iperm = inv_perm.get_data();
    for (size_type ind = 0; ind < inv_perm.get_num_elems(); ++ind) {
        iperm[perm[ind]] = ind;
    }
    column_permute_impl(&inv_perm, orig, column_permuted);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_COLUMN_PERMUTE_KERNEL);


template <typename ValueType, typename IndexType>
void inverse_column_permute(std::shared_ptr<const ReferenceExecutor> exec,
                            const Array<IndexType> *permutation_indices,
                            const matrix::Csr<ValueType, IndexType> *orig,
                            matrix::Csr<ValueType, IndexType> *column_permuted)
{
    column_permute_impl(permutation_indices, orig, column_permuted);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_INVERSE_COLUMN_PERMUTE_KERNEL);


template <typename ValueType, typename IndexType>
void calculate_nonzeros_per_row(std::shared_ptr<const ReferenceExecutor> exec,
                                const matrix::Csr<ValueType, IndexType> *source,
                                Array<size_type> *result)
{
    const auto row_ptrs = source->get_const_row_ptrs();
    auto row_nnz_val = result->get_data();
    for (size_type i = 0; i < result->get_num_elems(); i++) {
        row_nnz_val[i] = row_ptrs[i + 1] - row_ptrs[i];
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_CALCULATE_NONZEROS_PER_ROW_KERNEL);


template <typename ValueType, typename IndexType>
void sort_by_column_index(std::shared_ptr<const ReferenceExecutor> exec,
                          matrix::Csr<ValueType, IndexType> *to_sort)
{
    auto values = to_sort->get_values();
    auto row_ptrs = to_sort->get_row_ptrs();
    auto col_idxs = to_sort->get_col_idxs();
    const auto number_rows = to_sort->get_size()[0];
    for (size_type i = 0; i < number_rows; ++i) {
        auto start_row_idx = row_ptrs[i];
        auto row_nnz = row_ptrs[i + 1] - start_row_idx;
        auto helper = detail::IteratorFactory<IndexType, ValueType>(
            col_idxs + start_row_idx, values + start_row_idx, row_nnz);
        std::sort(helper.begin(), helper.end());
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_SORT_BY_COLUMN_INDEX);


template <typename ValueType, typename IndexType>
void is_sorted_by_column_index(
    std::shared_ptr<const ReferenceExecutor> exec,
    const matrix::Csr<ValueType, IndexType> *to_check, bool *is_sorted)
{
    const auto row_ptrs = to_check->get_const_row_ptrs();
    const auto col_idxs = to_check->get_const_col_idxs();
    const auto size = to_check->get_size();
    for (size_type i = 0; i < size[0]; ++i) {
        for (auto idx = row_ptrs[i] + 1; idx < row_ptrs[i + 1]; ++idx) {
            if (col_idxs[idx - 1] > col_idxs[idx]) {
                *is_sorted = false;
                return;
            }
        }
    }
    *is_sorted = true;
    return;
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_IS_SORTED_BY_COLUMN_INDEX);


template <typename ValueType, typename IndexType>
void extract_diag(std::shared_ptr<const ReferenceExecutor> exec,
                  const matrix::Csr<ValueType, IndexType> *source,
                  Array<ValueType> &diag)
{
    const auto dim_size = source->get_size();
    const auto row_ptrs = source->get_const_row_ptrs();
    const auto col_idxs = source->get_const_col_idxs();
    const auto vals = source->get_const_values();
    auto diag_val = diag.get_data();
    for (IndexType i = 0; i < dim_size[0]; i++) {
        for (auto j = row_ptrs[i]; j < row_ptrs[i + 1]; j++) {
            if (col_idxs[j] == i) {
                diag_val[i] = vals[j];
                break;
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_CSR_EXTRACT_DIAG);


template <typename ValueType, typename IndexType>
void find_strongest_neighbor(std::shared_ptr<const ReferenceExecutor> exec,
                             const matrix::Csr<ValueType, IndexType> *source,
                             const Array<ValueType> &diag,
                             Array<IndexType> &agg,
                             Array<IndexType> &strongest_neighbor)
{
    const auto row_ptrs = source->get_const_row_ptrs();
    const auto col_idxs = source->get_const_col_idxs();
    const auto vals = source->get_const_values();
    for (size_type row = 0; row < agg.get_num_elems(); row++) {
        auto max_weight_unagg = zero<remove_complex<ValueType>>();
        auto max_weight_agg = zero<remove_complex<ValueType>>();
        IndexType strongest_unagg = -1;
        IndexType strongest_agg = -1;
        if (agg.get_const_data()[row] == -1) {
            for (auto ind = row_ptrs[row]; ind < row_ptrs[row + 1]; ind++) {
                auto weight = zero<remove_complex<ValueType>>();
                auto col = col_idxs[ind];
                if (col == row) {
                    continue;
                }
                // Handle the unsymmetric values cases
                for (auto trans_ind = row_ptrs[col];
                     trans_ind < row_ptrs[col + 1]; trans_ind++) {
                    if (col_idxs[trans_ind] == row) {
                        weight = 0.5 * (abs(vals[ind]) + abs(vals[trans_ind])) /
                                 max(abs(diag.get_const_data()[row]),
                                     abs(diag.get_const_data()[col]));
                        break;
                    }
                }

                if (agg.get_const_data()[col] == -1 &&
                    (weight > max_weight_unagg ||
                     (weight == max_weight_unagg && col > strongest_unagg))) {
                    max_weight_unagg = weight;
                    strongest_unagg = col;
                } else if (agg.get_const_data()[col] != -1 &&
                           (weight > max_weight_agg ||
                            (weight == max_weight_agg &&
                             col > strongest_agg))) {
                    max_weight_agg = weight;
                    strongest_agg = col;
                }
            }

            if (strongest_unagg == -1 && strongest_agg != -1) {
                // all neighbor is agg, connect to the strongest agg
                agg.get_data()[row] = agg.get_data()[strongest_agg];
            } else if (strongest_unagg != -1) {
                // set the strongest neighbor in the unagg group
                strongest_neighbor.get_data()[row] = strongest_unagg;
            } else {
                // no neighbor
                strongest_neighbor.get_data()[row] = row;
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_FIND_STRONGEST_NEIGHBOR);


template <typename ValueType, typename IndexType>
void assign_to_exist_agg(std::shared_ptr<const ReferenceExecutor> exec,
                         const matrix::Csr<ValueType, IndexType> *source,
                         const Array<ValueType> &diag, Array<IndexType> &agg)
{
    const auto row_ptrs = source->get_const_row_ptrs();
    const auto col_idxs = source->get_const_col_idxs();
    const auto vals = source->get_const_values();
    const auto diag_vals = diag.get_const_data();
    auto max_weight_agg = zero<remove_complex<ValueType>>();

    for (IndexType row = 0; row < agg.get_num_elems(); row++) {
        if (agg.get_data()[row] != -1) {
            continue;
        }
        IndexType strongest_agg = -1;
        for (auto idx = row_ptrs[row]; idx < row_ptrs[row + 1]; idx++) {
            auto col = col_idxs[idx];
            if (col == row) {
                break;
            }
            auto weight =
                abs(vals[idx]) / max(abs(diag_vals[row]), abs(diag_vals[col]));
            if (agg.get_const_data()[col] != -1 &&
                (weight > max_weight_agg ||
                 (weight == max_weight_agg && col > strongest_agg))) {
                max_weight_agg = weight;
                strongest_agg = col;
            }
        }
        if (strongest_agg != -1) {
            agg.get_data()[row] = agg.get_const_data()[strongest_agg];
        } else {
            agg.get_data()[row] = row;
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_ASSIGN_TO_EXIST_AGG);


template <typename ValueType, typename IndexType>
void amgx_pgm_generate(std::shared_ptr<const ReferenceExecutor> exec,
                       const matrix::Csr<ValueType, IndexType> *source,
                       const Array<IndexType> &agg,
                       matrix::Csr<ValueType, IndexType> *coarse)
{
    // agg[i] -> I, agg[j] -> J
    const auto coarse_nrows = coarse->get_size()[0];
    const auto source_nrows = source->get_size()[0];
    const auto source_row_ptrs = source->get_const_row_ptrs();
    const auto source_col_idxs = source->get_const_col_idxs();
    const auto source_vals = source->get_const_values();
    vector<map<IndexType, ValueType>> row_list(
        source_nrows, map<IndexType, ValueType>{exec}, exec);
    for (size_type i = 0; i < source_nrows; i++) {
        IndexType row_idx = agg.get_const_data()[i];
        for (auto j = source_row_ptrs[i]; j < source_row_ptrs[i + 1]; j++) {
            const auto col = agg.get_const_data()[source_col_idxs[j]];
            const auto val = source_vals[j];
            row_list[row_idx][col] += val;
        }
    }
    auto coarse_row_ptrs = coarse->get_row_ptrs();
    for (size_type i = 0; i < coarse_nrows; i++) {
        coarse_row_ptrs[i] = row_list[i].size();
    }
    components::prefix_sum(exec, coarse_row_ptrs, coarse_nrows + 1);

    auto nnz = coarse_row_ptrs[coarse_nrows];
    matrix::CsrBuilder<ValueType, IndexType> coarse_builder{coarse};
    auto &coarse_col_idxs_array = coarse_builder.get_col_idx_array();
    auto &coarse_vals_array = coarse_builder.get_value_array();
    coarse_col_idxs_array.resize_and_reset(nnz);
    coarse_vals_array.resize_and_reset(nnz);
    auto coarse_col_idxs = coarse_col_idxs_array.get_data();
    auto coarse_vals = coarse_vals_array.get_data();

    for (size_type i = 0; i < coarse_nrows; i++) {
        auto ind = coarse_row_ptrs[i];
        for (auto pair : row_list[i]) {
            coarse_col_idxs[ind] = pair.first;
            coarse_vals[ind] = pair.second;
            ind++;
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_AMGX_PGM_GENERATE);


}  // namespace csr
}  // namespace reference
}  // namespace kernels
}  // namespace gko
