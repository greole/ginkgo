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

#include "core/solver/cg_kernels.hpp"


#include <omp.h>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>


namespace gko {
namespace kernels {
namespace omp {
/**
 * @brief The CG solver namespace.
 *
 * @ingroup cg
 */
namespace cg {


template <typename ValueType>
void initialize(std::shared_ptr<const OmpExecutor> exec,
                const matrix::Dense<ValueType> *b, matrix::Dense<ValueType> *r,
                matrix::Dense<ValueType> *z, matrix::Dense<ValueType> *p,
                matrix::Dense<ValueType> *q, matrix::Dense<ValueType> *prev_rho,
                matrix::Dense<ValueType> *rho,
                Array<stopping_status> *stop_status)
{
#pragma omp parallel for
    for (size_type j = 0; j < b->get_size()[1]; ++j) {
        rho->at(j) = zero<ValueType>();
        prev_rho->at(j) = one<ValueType>();
        stop_status->get_data()[j].reset();
    }
#pragma omp parallel for
    for (size_type i = 0; i < b->get_size()[0]; ++i) {
        for (size_type j = 0; j < b->get_size()[1]; ++j) {
            r->at(i, j) = b->at(i, j);
            z->at(i, j) = p->at(i, j) = q->at(i, j) = zero<ValueType>();
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_CG_INITIALIZE_KERNEL);


template <typename ValueType>
struct Functor {
    std::shared_ptr<const OmpExecutor> exec;
    matrix::Dense<ValueType> *p;
    const matrix::Dense<ValueType> *z;
    const matrix::Dense<ValueType> *rho;
    const matrix::Dense<ValueType> *prev_rho;
    const Array<stopping_status> *stop_status;

    typedef ValueType value_type;

    Functor(std::shared_ptr<const OmpExecutor> exec,
            matrix::Dense<ValueType> *p, const matrix::Dense<ValueType> *z,
            const matrix::Dense<ValueType> *rho,
            const matrix::Dense<ValueType> *prev_rho,
            const Array<stopping_status> *stop_status)
        : exec(exec),
          p(p),
          z(z),
          rho(rho),
          prev_rho(prev_rho),
          stop_status(stop_status)
    {}

    virtual void operator()(size_type, size_type) {}
};


template <typename ValueType>
struct CG_inner_step1 : public Functor<ValueType> {
    using Functor<ValueType>::Functor;
    void operator()(size_type i, size_type j)
    {
        auto tmp = this->rho->at(j) / this->prev_rho->at(j);
        this->p->at(i, j) = this->z->at(i, j) + tmp * this->p->at(i, j);
    }
};


// TODO Replace by macro, instantiate for all data types
template void Functor<float>::operator()(size_type, size_type);

template <typename Functor>
void step_1_impl_tensor(Functor func)
{
#pragma omp parallel for
    for (size_type i = 0; i < func.p->get_size()[0]; ++i) {
        for (size_type j = 0; j < func.p->get_size()[1]; ++j) {
            if (func.stop_status->get_const_data()[j].has_stopped()) {
                continue;
            }
            if (func.prev_rho->at(j) == zero<typename Functor::value_type>()) {
                func.p->at(i, j) = func.z->at(i, j);
            } else {
                func(i, j);
            }
        }
    }
}

template <typename Functor>
void step_1_impl_matrix(Functor func)
{
#pragma omp parallel for
    for (size_type i = 0; i < func.p->get_size()[0]; ++i) {
        func(i, 0);
    }
}

template <typename Functor>
void dispatch(Functor inner)
{
    if (inner.p->get_size()[1] > 1) {
        step_1_impl_tensor(inner);
    } else {
        step_1_impl_matrix(inner);
    }
}

template <typename ValueType>
void step_1(std::shared_ptr<const OmpExecutor> exec,
            matrix::Dense<ValueType> *p, const matrix::Dense<ValueType> *z,
            const matrix::Dense<ValueType> *rho,
            const matrix::Dense<ValueType> *prev_rho,
            const Array<stopping_status> *stop_status)
{
    auto functor =
        CG_inner_step1<ValueType>(exec, p, z, rho, prev_rho, stop_status);
    dispatch(functor);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_CG_STEP_1_KERNEL);


template <typename ValueType>
void step_2(std::shared_ptr<const OmpExecutor> exec,
            matrix::Dense<ValueType> *x, matrix::Dense<ValueType> *r,
            const matrix::Dense<ValueType> *p,
            const matrix::Dense<ValueType> *q,
            const matrix::Dense<ValueType> *beta,
            const matrix::Dense<ValueType> *rho,
            const Array<stopping_status> *stop_status)
{
#pragma omp parallel for
    for (size_type j = 0; j < x->get_size()[1]; ++j) {
        if (stop_status->get_const_data()[j].has_stopped()) {
            continue;
        }
        for (size_type i = 0; i < x->get_size()[0]; ++i) {
            if (beta->at(j) != zero<ValueType>()) {
                auto tmp = rho->at(j) / beta->at(j);
                x->at(i, j) += tmp * p->at(i, j);
                r->at(i, j) -= tmp * q->at(i, j);
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_CG_STEP_2_KERNEL);


}  // namespace cg
}  // namespace omp
}  // namespace kernels
}  // namespace gko
