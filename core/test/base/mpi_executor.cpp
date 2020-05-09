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


#include <type_traits>


#include <gtest/gtest.h>


#include <core/test/utils.hpp>
#include <ginkgo/core/base/exception.hpp>


namespace {


// using mpi = std::shared_ptr<gko::Executor>;


class ExampleOperation : public gko::Operation {
public:
    explicit ExampleOperation(int &val) : value(val) {}
    void run(std::shared_ptr<const gko::OmpExecutor>) const override
    {
        value = 1;
    }
    void run(std::shared_ptr<const gko::MpiExecutor>) const override
    {
        value = 2;
    }
    void run(std::shared_ptr<const gko::ReferenceExecutor>) const override
    {
        value = 3;
    }
    void run(std::shared_ptr<const gko::CudaExecutor>) const override
    {
        value = 4;
    }

    int &value;
};

class MPITest : public ::testing::Test {
protected:
    using exec = gko::MpiExecutor;
    void SetUp() { mpi = gko::MpiExecutor::create(); }

    // void TearDown() { gko::MpiExecutor::destroy(); }
    void TearDown() {}

    std::shared_ptr<gko::Executor> mpi;
    // std::shared_ptr<gko::MpiExecutor> mpi;
};

TEST(MpiExecutor, IsItsOwnMaster)
{
    auto mpi = gko::MpiExecutor::create();

    ASSERT_EQ(mpi, mpi->get_master());
}


TEST_F(MPITest, RunsCorrectOperation)
{
    int value = 0;
    auto mpi = gko::MpiExecutor::create();

    mpi->run(ExampleOperation(value));
    ASSERT_EQ(2, value);
}


TEST_F(MPITest, RunsCorrectLambdaOperation)
{
    int value = 0;
    auto omp_lambda = [&value]() { value = 1; };
    auto mpi_lambda = [&value]() { value = 2; };
    auto cuda_lambda = [&value]() { value = 4; };
    auto hip_lambda = [&value]() { value = 4; };
    auto mpi = gko::MpiExecutor::create();

    mpi->run(omp_lambda, mpi_lambda, cuda_lambda, hip_lambda);
    ASSERT_EQ(2, value);
}


}  // namespace
