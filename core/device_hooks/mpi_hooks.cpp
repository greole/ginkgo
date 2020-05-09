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

#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/version.hpp>


namespace gko {


version version_info::get_mpi_version() noexcept
{
    // We just return 1.0.0 with a special "not compiled" tag in placeholder
    // modules.
    return {1, 0, 0, "not compiled"};
}

void MpiExecutor::mpi_init() {}

void MpiExecutor::create_sub_executors(
    std::vector<std::string> &sub_exec_list,
    std::shared_ptr<gko::Executor> &sub_executor)
{}

int MpiExecutor::get_num_ranks() { return 0; }

// int MpiExecutor::get_num_gpus() { return 0; }

std::shared_ptr<MpiExecutor> MpiExecutor::create(
    std::initializer_list<std::string> sub_exec_list, int num_args, char **args)
{
    return std::shared_ptr<MpiExecutor>(
        new MpiExecutor(sub_exec_list, num_args, args));
}

// void MpiExecutor::run(const Operation &op) const
// {
//     op.run(
//         std::static_pointer_cast<const
//         MpiExecutor>(this->shared_from_this()));
// }

std::shared_ptr<MpiExecutor> MpiExecutor::create()
{
    int num_args = 0;
    char **args;
    return MpiExecutor::create({}, num_args, args);
}


std::string MpiError::get_error(int64)
{
    return "ginkgo MPI module is not compiled";
}


bool MpiExecutor::is_finalized() GKO_NOT_COMPILED(mpi);


bool MpiExecutor::is_initialized() GKO_NOT_COMPILED(mpi);


void MpiExecutor::destroy() GKO_NOT_COMPILED(mpi);


// void MpiExecutor::synchronize_communicator(
//     gko::MpiExecutor::handle_manager<MpiContext> comm) const
//     GKO_NOT_COMPILED(mpi);

void MpiExecutor::synchronize() const GKO_NOT_COMPILED(mpi);

}  // namespace gko


#define GKO_HOOK_MODULE mpi
#include "core/device_hooks/common_kernels.inc.cpp"
#undef GKO_HOOK_MODULE
