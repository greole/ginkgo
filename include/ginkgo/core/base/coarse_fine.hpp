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

#ifndef GKO_CORE_BASE_COARSEFINE_HPP_
#define GKO_CORE_BASE_COARSEFINE_HPP_


#include <functional>
#include <memory>

#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/matrix/dense.hpp>


namespace gko {


/**
 * The CoarseFine class can be used to construct a LinOp to represent the
 * operation ``. This operator adds a
 * movement along a direction constructed by `basis` and `projector` on the
 * LinOp. `projector` gives the coefficient of `basis` to decide the direction.
 *
 * @ingroup LinOp
 */

class CoarseFine : public EnableLinOp<CoarseFine> {
    friend class EnablePolymorphicObject<CoarseFine, LinOp>;
    friend class EnableCreateMethod<CoarseFine>;

public:
    // Return R * A * P
    virtual std::shared_ptr<const LinOp> get_coarse_operator() const
    {
        return coarse_;
    }
    // x = R * b
    void restrict_apply(const LinOp *b, LinOp *x) const
    {
        GKO_ASSERT_EQ(fine_dim_, b->get_size()[0]);
        GKO_ASSERT_EQ(coarse_dim_, x->get_size()[0]);
        GKO_ASSERT_EQUAL_COLS(b, x);
        auto exec = this->get_executor();
        this->restrict_apply_(make_temporary_clone(exec, b).get(),
                              make_temporary_clone(exec, x).get());
    };
    // x = x + P * b
    void prolongate_applyadd(const LinOp *b, LinOp *x) const
    {
        GKO_ASSERT_EQ(coarse_dim_, b->get_size()[0]);
        GKO_ASSERT_EQ(fine_dim_, x->get_size()[0]);
        GKO_ASSERT_EQUAL_COLS(b, x);
        auto exec = this->get_executor();
        this->prolongate_applyadd_(make_temporary_clone(exec, b).get(),
                                   make_temporary_clone(exec, x).get());
    }

protected:
    void apply_impl(const LinOp *b, LinOp *x) const { coarse_->apply(b, x); }

    void apply_impl(const LinOp *alpha, const LinOp *b, const LinOp *beta,
                    LinOp *x) const
    {
        coarse_->apply(alpha, b, beta, x);
    }
    // virtual void restrict_apply_impl(const LinOp *b, LinOp *x) const = 0;
    // virtual void prolongate_applyadd_impl(const LinOp *b, LinOp *x) const =
    // 0;
    void set_coarse_fine(
        std::shared_ptr<const LinOp> coarse,
        std::function<void(const LinOp *, LinOp *)> restrict_func,
        std::function<void(const LinOp *, LinOp *)> prolongate_func,
        size_type fine_dim)
    {
        coarse_ = coarse;
        fine_dim_ = fine_dim;
        coarse_dim_ = coarse->get_size()[0];
        restrict_apply_ = restrict_func;
        prolongate_applyadd_ = prolongate_func;
    }
    explicit CoarseFine(std::shared_ptr<const Executor> exec)
        : EnableLinOp<CoarseFine>(exec)
    {}

private:
    std::function<void(const LinOp *, LinOp *)> restrict_apply_;
    std::function<void(const LinOp *, LinOp *)> prolongate_applyadd_;
    std::shared_ptr<const LinOp> coarse_{};
    size_type fine_dim_;
    size_type coarse_dim_;
};

template <typename ValueType, typename IndexType>
class AmgxPgmOp {
public:
    virtual void extract_diag(Array<ValueType> &diag) const = 0;
    virtual void find_strongest_neighbor(
        const Array<ValueType> &diag, Array<IndexType> &agg,
        Array<IndexType> &strongest_neighbor) const = 0;
    virtual void assign_to_exist_agg(const Array<ValueType> &diag,
                                     Array<IndexType> &agg) const = 0;
    virtual std::unique_ptr<LinOp> amgx_pgm_generate(
        const Array<IndexType> &agg) const = 0;
};


}  // namespace gko


#endif  // GKO_CORE_BASE_COARSEFINE_HPP_
