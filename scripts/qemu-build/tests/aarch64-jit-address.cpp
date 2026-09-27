// SPDX-License-Identifier: GPL-2.0-or-later
// Exercise a JIT constant pool beyond AArch64 ADRP's signed 4 GiB range.
#include <llvm/ExecutionEngine/ExecutionEngine.h>
#include <llvm/ExecutionEngine/MCJIT.h>
#include <llvm/ExecutionEngine/RTDyldMemoryManager.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/Support/Memory.h>
#include <llvm/Support/TargetSelect.h>
#include "lp_bld_jit_memory.h"

#include <sys/mman.h>

#include <array>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>

namespace {
[[noreturn]] void fail(const char *message)
{
    std::perror(message);
    std::exit(1);
}

class DistantSections final : public llvm::RTDyldMemoryManager {
    static constexpr size_t section_size = 64 * 1024;
    static constexpr size_t distance = size_t{6} << 30;
    static constexpr size_t reservation_size = distance + section_size;
    uint8_t *base;
    size_t code_used = 0;
    size_t data_used = 0;

    uint8_t *allocate(size_t start, size_t &used, uintptr_t size, unsigned alignment)
    {
        size_t align = alignment ? alignment : 16;
        size_t offset = (used + align - 1) & ~(align - 1);
        if (offset > section_size || size > section_size - offset) {
            std::fputs("test section exhausted\n", stderr);
            std::exit(1);
        }
        used = offset + size;
        return base + start + offset;
    }

public:
    DistantSections()
    {
        void *memory = mmap(nullptr, reservation_size, PROT_NONE,
                            MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (memory == MAP_FAILED)
            fail("mmap");
        base = static_cast<uint8_t *>(memory);
        if (mprotect(base, section_size, PROT_READ | PROT_WRITE) ||
            mprotect(base + distance, section_size, PROT_READ | PROT_WRITE))
            fail("mprotect");
    }

    ~DistantSections() override
    {
        if (munmap(base, reservation_size))
            fail("munmap");
    }

    uint8_t *allocateCodeSection(uintptr_t size, unsigned alignment, unsigned,
                                 llvm::StringRef) override
    {
        return allocate(0, code_used, size, alignment);
    }

    uint8_t *allocateDataSection(uintptr_t size, unsigned alignment, unsigned,
                                 llvm::StringRef, bool) override
    {
        return allocate(distance, data_used, size, alignment);
    }

    bool finalizeMemory(std::string *error) override
    {
        if (mprotect(base, section_size, PROT_READ | PROT_EXEC)) {
            if (error)
                *error = "could not protect generated code";
            return true;
        }
        llvm::sys::Memory::InvalidateInstructionCache(base, code_used);
        return false;
    }

    // This test's generated function is nounwind and never throws.
    void registerEHFrames(uint8_t *, uint64_t, size_t) override {}
    void deregisterEHFrames() override {}
};

bool check_reservations()
{
    struct Object {
        uint8_t *code;
        uint8_t *ro;
        uint8_t *rw;
    };
    std::array<Object, 3> objects;
    {
        LPContiguousMemoryManager manager;
        for (unsigned i = 0; i < objects.size(); ++i) {
            unsigned alignment = 16u << (6 * i);
            manager.reserveAllocationSpace(64, llvm::Align(alignment),
                                           64, llvm::Align(alignment),
                                           64, llvm::Align(alignment));
            auto &o = objects[i];
            o.code = manager.allocateCodeSection(8, alignment, 0, "code");
            o.ro = manager.allocateDataSection(64, alignment, 1, "ro", true);
            o.rw = manager.allocateDataSection(64, alignment, 2, "rw", false);
            const uint32_t instructions[] = {0x52800000u | ((i + 1) << 5), 0xd65f03c0u};
            std::memcpy(o.code, instructions, sizeof(instructions));
            std::memset(o.ro, i + 1, 64);
            for (auto *p : {o.code, o.ro, o.rw}) {
                if (reinterpret_cast<uintptr_t>(p) % alignment) {
                    std::fputs("section alignment was not preserved\n", stderr);
                    return false;
                }
            }
            auto span = reinterpret_cast<uintptr_t>(o.rw) + 64 -
                        reinterpret_cast<uintptr_t>(o.code);
            if (span > INT32_MAX) {
                std::fputs("reserved object exceeds relocation range\n", stderr);
                return false;
            }
            std::string error;
            if (manager.finalizeMemory(&error)) {
                std::fprintf(stderr, "finalization failed: %s\n", error.c_str());
                return false;
            }
            for (unsigned j = 0; j <= i; ++j) {
                auto &old = objects[j];
                auto function = reinterpret_cast<uint32_t (*)()>(old.code);
                std::memset(old.rw, i + 1, 64);
                if (function() != j + 1 || old.ro[63] != j + 1 ||
                    old.rw[63] != i + 1) {
                    std::fputs("finalized object did not survive later reservations\n", stderr);
                    return false;
                }
            }
        }
    }
    const uintptr_t page = llvm::sys::Process::getPageSizeEstimate();
    for (auto &o : objects) {
        for (auto *p : {o.code, o.ro, o.rw}) {
            auto address = reinterpret_cast<uintptr_t>(p) & ~(page - 1);
            unsigned char resident;
            errno = 0;
            if (mincore(reinterpret_cast<void *>(address), page, &resident) != -1 ||
                errno != ENOMEM) {
                std::fputs("destroyed memory manager retained a section mapping\n", stderr);
                return false;
            }
        }
    }
    std::puts("repeated reservation, alignment, execution and release checks passed");
    return true;
}
} // namespace

int main(int argc, char **argv)
{
    if (argc != 2 || (std::string(argv[1]) != "small" &&
                      std::string(argv[1]) != "reserved" &&
                      std::string(argv[1]) != "large-globalisel")) {
        std::fputs("usage: aarch64-jit-address small|reserved|large-globalisel\n", stderr);
        return 2;
    }
    if (llvm::InitializeNativeTarget() || llvm::InitializeNativeTargetAsmPrinter()) {
        std::fputs("native LLVM target initialization failed\n", stderr);
        return 1;
    }
    if (std::string(argv[1]) == "reserved" && !check_reservations())
        return 1;

    llvm::LLVMContext context;
    auto module = std::make_unique<llvm::Module>("distant-constant-pool", context);
    llvm::IRBuilder<> ir(context);
    const uint32_t values[] = {0x12345678, 0x89abcdef, 0x31415926, 0x27182818};
    auto *initializer = llvm::ConstantDataArray::get(context, values);
    auto *table = new llvm::GlobalVariable(
        *module, initializer->getType(), true, llvm::GlobalValue::InternalLinkage,
        initializer, "table");
    auto *function = llvm::Function::Create(
        llvm::FunctionType::get(ir.getInt32Ty(), {ir.getInt32Ty()}, false),
        llvm::GlobalValue::ExternalLinkage, "lookup", *module);
    function->addFnAttr(llvm::Attribute::NoUnwind);
    ir.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", function));
    auto *index = ir.CreateAnd(function->getArg(0), ir.getInt32(3));
    auto *address = ir.CreateInBoundsGEP(initializer->getType(), table,
                                       {ir.getInt32(0), index});
    auto *value = ir.CreateLoad(ir.getInt32Ty(), address);
    value->setVolatile(true);
    ir.CreateRet(value);

    auto *vector_function = llvm::Function::Create(
        llvm::FunctionType::get(ir.getVoidTy(),
                               {ir.getPtrTy(), ir.getInt32Ty()}, false),
        llvm::GlobalValue::ExternalLinkage, "vector_constants", *module);
    vector_function->addFnAttr(llvm::Attribute::NoUnwind);
    ir.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", vector_function));
    auto *scaled = ir.CreateMul(llvm::ConstantDataVector::get(context, values),
                               ir.CreateVectorSplat(4, vector_function->getArg(1)));
    ir.CreateStore(scaled, vector_function->getArg(0));
    ir.CreateRetVoid();

    auto *scatter_function = llvm::Function::Create(
        vector_function->getFunctionType(), llvm::GlobalValue::ExternalLinkage,
        "scatter", *module);
    scatter_function->addFnAttr(llvm::Attribute::NoUnwind);
    ir.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", scatter_function));
    const uint32_t indices[] = {0, 1, 2, 3};
    auto *pointers = ir.CreateGEP(ir.getInt32Ty(), scatter_function->getArg(0),
                                llvm::ConstantDataVector::get(context, indices));
    auto *mask = ir.CreateICmpULE(
        llvm::ConstantDataVector::get(context, indices),
        ir.CreateVectorSplat(4, ir.CreateAnd(scatter_function->getArg(1), ir.getInt32(3))));
    ir.CreateMaskedScatter(llvm::ConstantDataVector::get(context, values),
                           pointers, llvm::Align(4), mask);
    ir.CreateRetVoid();

    std::string error;
    std::unique_ptr<llvm::RTDyldMemoryManager> manager;
    if (std::string(argv[1]) == "reserved")
        manager = std::make_unique<LPContiguousMemoryManager>();
    else
        manager = std::make_unique<DistantSections>();
    llvm::EngineBuilder builder(std::move(module));
    builder.setEngineKind(llvm::EngineKind::JIT)
        .setErrorStr(&error)
        .setOptLevel(llvm::CodeGenOptLevel::None)
        .setCodeModel(std::string(argv[1]) == "large-globalisel" ? llvm::CodeModel::Large
                                                              : llvm::CodeModel::Small)
        .setMCJITMemoryManager(std::move(manager));
    std::unique_ptr<llvm::ExecutionEngine> engine(builder.create());
    if (!engine) {
        std::fprintf(stderr, "JIT creation failed: %s\n", error.c_str());
        return 1;
    }
    engine->finalizeObject();
    auto lookup = reinterpret_cast<uint32_t (*)(uint32_t)>(
        engine->getFunctionAddress("lookup"));
    if (!lookup) {
        std::fputs("JIT function missing\n", stderr);
        return 1;
    }
    for (uint32_t i = 0; i < 64; ++i) {
        if (lookup(i) != values[i & 3]) {
            std::fprintf(stderr, "incorrect constant at index %u\n", i);
            return 1;
        }
    }
    auto scatter = reinterpret_cast<void (*)(uint32_t *, uint32_t)>(
        engine->getFunctionAddress("scatter"));
    if (!scatter) {
        std::fputs("JIT scatter function missing\n", stderr);
        return 1;
    }
    for (uint32_t i = 0; i < 64; ++i) {
        alignas(16) uint32_t result[4] = {};
        scatter(result, i);
        for (uint32_t lane = 0; lane < 4; ++lane) {
            if (result[lane] != (lane <= (i & 3) ? values[lane] : 0)) {
                std::fprintf(stderr, "incorrect scatter lane at index %u\n", i);
                return 1;
            }
        }
    }
    auto vector_constants = reinterpret_cast<void (*)(uint32_t *, uint32_t)>(
        engine->getFunctionAddress("vector_constants"));
    if (!vector_constants) {
        std::fputs("JIT vector function missing\n", stderr);
        return 1;
    }
    for (uint32_t i = 0; i < 64; ++i) {
        alignas(16) uint32_t result[4];
        vector_constants(result, i);
        for (uint32_t lane = 0; lane < 4; ++lane) {
            if (result[lane] != values[lane] * i) {
                std::fprintf(stderr, "incorrect vector constant at index %u\n", i);
                return 1;
            }
        }
    }
    auto function_address = engine->getFunctionAddress("lookup");
    auto data_address = engine->getGlobalValueAddress("table");
    auto distance = function_address > data_address ? function_address - data_address
                                                    : data_address - function_address;
    if (std::string(argv[1]) == "reserved" && distance > INT32_MAX) {
        std::fputs("code and data exceed relocation range\n", stderr);
        return 1;
    }
    std::printf("64 global, vector and scatter cases passed; section distance = %llu bytes\n",
                static_cast<unsigned long long>(distance));
}
