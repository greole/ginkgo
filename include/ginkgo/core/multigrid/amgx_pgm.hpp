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

#ifndef GKO_CORE_MULTIGRID_AMGX_PGM_HPP_
#define GKO_CORE_MULTIGRID_AMGX_PGM_HPP_


#include <vector>


#include <ginkgo/core/base/coarse_fine.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/csr.hpp>


namespace gko {
namespace multigrid {


/**
 * Iterative refinement (IR) is an iterative method that uses another coarse
 * method to approximate the error of the current solution via the current
 * residual.
 *
 * For any approximation of the solution `solution` to the system `Ax = b`, the
 * residual is defined as: `residual = b - A solution`. The error in
 * `solution`,  `e = x - solution` (with `x` being the exact solution) can be
 * obtained as the solution to the residual equation `Ae = residual`, since `A e
 * = Ax - A solution = b - A solution = residual`. Then, the real solution is
 * computed as `x = solution + e`. Instead of accurately solving the residual
 * equation `Ae = residual`, the solution of the system `e` can be approximated
 * to obtain the approximation `error` using a coarse method `multigrid`, which
 * is used to update `solution`, and the entire process is repeated with the
 * updated `solution`.  This yields the iterative refinement method:
 *
 * ```
 * solution = initial_guess
 * while not converged:
 *     residual = b - A solution
 *     error = multigrid(A, residual)
 *     solution = solution + error
 * ```
 *
 * Assuming that `multigrid` has accuracy `c`, i.e., `| e - error | <= c | e |`,
 * iterative refinement will converge with a convergence rate of `c`. Indeed,
 * from `e - error = x - solution - error = x - solution*` (where `solution*`
 * denotes the value stored in `solution` after the update) and `e = inv(A)
 * residual = inv(A)b - inv(A) A solution = x - solution` it follows that | x -
 * solution* | <= c | x - solution |.
 *
 * Unless otherwise specified via the `multigrid` factory parameter, this
 * implementation uses the identity operator (i.e. the multigrid that
 * approximates the solution of a system Ax = b by setting x := b) as the
 * default inner multigrid. Such a setting results in a relaxation method known
 * as the Richardson iteration with parameter 1, which is guaranteed to converge
 * for matrices whose spectrum is strictly contained within the unit disc around
 * 1 (i.e., all its eigenvalues `lambda` have to satisfy the equation `|lambda -
 * 1| < 1).
 *
 * @tparam ValueType  precision of matrix elements
 *
 * @ingroup multigrids
 * @ingroup LinOp
 */
template <typename ValueType = default_precision, typename IndexType = int32>
class AmgxPgm : public CoarseFine {
public:
    using value_type = ValueType;
    using index_type = IndexType;
    using matrix_type = matrix::Csr<ValueType, IndexType>;
    /**
     * Returns the system operator (matrix) of the linear system.
     *
     * @return the system operator (matrix)
     */
    std::shared_ptr<const LinOp> get_system_matrix() const
    {
        return system_matrix_;
    }


    IndexType *get_agg() noexcept { return agg_.get_data(); }


    const IndexType *get_const_agg() const noexcept
    {
        return agg_.get_const_data();
    }


    GKO_CREATE_FACTORY_PARAMETERS(parameters, Factory)
    {
        unsigned GKO_FACTORY_PARAMETER(max_iterations, 15);
        double GKO_FACTORY_PARAMETER(max_unassigned_percentage, 0.05);
    };
    GKO_ENABLE_LIN_OP_FACTORY(AmgxPgm, parameters, Factory);
    GKO_ENABLE_BUILD_METHOD(Factory);

protected:
    void restrict_apply_impl(const LinOp *b, LinOp *x) const;
    void prolongate_applyadd_impl(const LinOp *b, LinOp *x) const;

    explicit AmgxPgm(const Factory *factory,
                     std::shared_ptr<const LinOp> system_matrix)
        : CoarseFine(factory->get_executor()),
          parameters_{factory->get_parameters()},
          system_matrix_{std::move(system_matrix)},
          agg_(factory->get_executor(), system_matrix_->get_size()[0]),
          diag_(factory->get_executor(), system_matrix_->get_size()[0])
    {
        this->generate();
    }

    void generate();

private:
    std::shared_ptr<const LinOp> system_matrix_{};
    Array<ValueType> diag_;
    Array<IndexType> agg_;
};


}  // namespace multigrid
}  // namespace gko


#endif  // GKO_CORE_MULTIGRID_AMGX_PGM_HPP_
