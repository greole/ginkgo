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

#include "core/multigrid/amgx_pgm_kernels.hpp"


#include <memory>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/multigrid/amgx_pgm.hpp>
#include "core/components/prefix_sum.hpp"

namespace gko {
namespace kernels {
namespace reference {
/**
 * @brief The AMGX_PGM solver namespace.
 *
 * @ingroup amgx_pgm
 */
namespace amgx_pgm {


template <typename ValueType, typename IndexType>
void restrict_apply(std::shared_ptr<const ReferenceExecutor> exec,
                    const Array<IndexType> &agg,
                    const matrix::Dense<ValueType> *b,
                    matrix::Dense<ValueType> *x)
{
    auto x_vals = x->get_values();
    const auto x_stride = x->get_stride();
    auto x_dim = x->get_size();
    for (size_type i = 0; i < x_dim[0]; i++) {
        for (size_type j = 0; j < x_dim[1]; j++) {
            x->at(i, j) = zero<ValueType>();
        }
    }
    for (size_type i = 0; i < agg.get_num_elems(); i++) {
        const auto x_row = agg.get_const_data()[i];
        for (size_type j = 0; j < x_dim[1]; j++) {
            x->at(x_row, j) += b->at(i, j);
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_AMGX_PGM_RESTRICT_APPLY_KERNEL);


template <typename ValueType, typename IndexType>
void prolongate_applyadd(std::shared_ptr<const ReferenceExecutor> exec,
                         const Array<IndexType> &agg,
                         const matrix::Dense<ValueType> *b,
                         matrix::Dense<ValueType> *x)
{
    auto x_vals = x->get_values();
    const auto x_stride = x->get_stride();
    auto x_dim = x->get_size();
    for (size_type i = 0; i < agg.get_num_elems(); i++) {
        const auto b_row = agg.get_const_data()[i];
        for (size_type j = 0; j < x_dim[1]; j++) {
            x->at(i, j) += b->at(b_row, j);
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_AMGX_PGM_PROLONGATE_APPLY_KERNEL);


template <typename IndexType>
void match_edge(std::shared_ptr<const ReferenceExecutor> exec,
                const Array<IndexType> &strongest_neighbor,
                Array<IndexType> &agg)
{
    auto agg_vals = agg.get_data();
    auto strongest_neighbor_vals = strongest_neighbor.get_const_data();
    for (size_type i = 0; i < agg.get_num_elems(); i++) {
        if (agg_vals[i] == -1) {
            auto neighbor = strongest_neighbor_vals[i];
            if (neighbor != -1 && strongest_neighbor_vals[neighbor] == i) {
                agg_vals[i] = i;
                agg_vals[neighbor] = i;
                // Use the smaller index as agg point
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_AMGX_PGM_MATCH_EDGE_KERNEL);


template <typename IndexType>
void count_unagg(std::shared_ptr<const ReferenceExecutor> exec,
                 const Array<IndexType> &agg, IndexType *num_unagg)
{
    IndexType unagg = 0;
    for (size_type i = 0; i < agg.get_num_elems(); i++) {
        unagg += (agg.get_const_data()[i] == -1);
    }
    *num_unagg = unagg;
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_AMGX_PGM_COUNT_UNAGG_KERNEL);


template <typename IndexType>
void renumber(std::shared_ptr<const ReferenceExecutor> exec,
              Array<IndexType> &agg, size_type *num_agg)
{
    const auto num = agg.get_num_elems();
    Array<IndexType> agg_map(exec, num);
    auto agg_vals = agg.get_data();
    auto agg_map_vals = agg_map.get_data();
    for (size_type i = 0; i < num; i++) {
        agg_map_vals[agg_vals[i]] = 1;
    }
    components::prefix_sum(exec, agg_map_vals, num);
    for (size_type i = 0; i < num; i++) {
        agg_vals[i] = agg_map_vals[agg_vals[i]];
    }
    *num_agg = agg_map_vals[num - 1];
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_AMGX_PGM_RENUMBER_KERNEL);


}  // namespace amgx_pgm
}  // namespace reference
}  // namespace kernels
}  // namespace gko
