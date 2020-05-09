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

#include <ginkgo/core/base/executor.hpp>


#include <iostream>
#include <map>


#include <mpi.h>


#include <ginkgo/config.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>


#include "mpi/base/mpi_bindings.hpp"


namespace gko {


void MpiExecutor::mpi_init()
{
    auto flag = MpiExecutor::is_initialized();
    if (!flag) {
        GKO_ASSERT_NO_MPI_ERRORS(MPI_Init_thread(
            &(this->num_args_), &(this->args_), this->required_thread_support_,
            &(this->provided_thread_support_)));
    }
}


void MpiExecutor::create_sub_executors(
    std::vector<std::string> &sub_exec_list,
    std::shared_ptr<gko::Executor> &sub_executor)
{
    auto num_gpus = this->get_num_gpus();
    int dev_id = 0;
    for (auto i = 0; i < sub_exec_list.size(); ++i) {
        if (sub_exec_list[i] == "omp") {
            sub_executor = gko::OmpExecutor::create();
        }
        if (sub_exec_list[i] == "reference") {
            sub_executor = gko::ReferenceExecutor::create();
        }
        if (sub_exec_list[i] == "cuda" && num_gpus > 0 && dev_id < num_gpus) {
            sub_executor =
                gko::CudaExecutor::create(dev_id, gko::OmpExecutor::create());
            dev_id++;
        }
        if (sub_exec_list[i] == "hip" && num_gpus > 0 && dev_id < num_gpus) {
            sub_executor =
                gko::HipExecutor::create(dev_id, gko::OmpExecutor::create());
            dev_id++;
        }
    }
}

// void MpiExecutor::synchronize_communicator(
//     gko::MpiExecutor::handle_manager<MpiContext> comm) const
// {
//     // kernels::mpi::synchronize(comm);
//     GKO_ASSERT_NO_MPI_ERRORS(MPI_Barrier(comm));
// }


void MpiExecutor::synchronize() const
{
    GKO_ASSERT_NO_MPI_ERRORS(MPI_Barrier(MPI_COMM_WORLD));
}


int MpiExecutor::get_my_rank()
{
    auto my_rank = 0;
    GKO_ASSERT_NO_MPI_ERRORS(MPI_Comm_rank(MPI_COMM_WORLD, &my_rank));
    return my_rank;
}


int MpiExecutor::get_num_ranks()
{
    int size = 1;
    GKO_ASSERT_NO_MPI_ERRORS(MPI_Comm_size(MPI_COMM_WORLD, &size));
    return size;
}


std::shared_ptr<MpiExecutor> MpiExecutor::create(
    std::initializer_list<std::string> sub_exec_list, int num_args, char **args)
{
    return std::shared_ptr<MpiExecutor>(
        new MpiExecutor(sub_exec_list, num_args, args),
        [](MpiExecutor *exec) { delete exec; });
}


std::shared_ptr<MpiExecutor> MpiExecutor::create()
{
    int num_args = 0;
    char **args;
    return MpiExecutor::create({"reference"}, num_args, args);
}


bool MpiExecutor::is_initialized()
{
    int flag = 0;
    GKO_ASSERT_NO_MPI_ERRORS(MPI_Initialized(&flag));
    return flag;
}


bool MpiExecutor::is_finalized()
{
    int flag = 0;
    GKO_ASSERT_NO_MPI_ERRORS(MPI_Finalized(&flag));
    return flag;
}


void MpiExecutor::destroy()
{
    auto flag = MpiExecutor::is_finalized();
    if (!flag) {
        GKO_ASSERT_NO_MPI_ERRORS(MPI_Finalize());
    }
}


// MpiContext MpiExecutor::create_comms()
// {
//     auto mpi_comm_ = handle_manager<MpiContext>(
//         kernels::mpi::create_comms(comm_in, color, key),
//         [](Mpi_Comm comm) { kernels::mpi::free_comms(comm); });
//     return mpi_comm_;
// }


}  // namespace gko
