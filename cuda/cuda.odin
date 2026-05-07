package cuda

import "core:io"
import "core:fmt"
import "core:c"
import "core:sys/linux"
import "core:reflect"
import "base:runtime"

MAX_CLOCK_TS :: 3

cuComplex :: [2]f32
cuDoubleComplex :: [2]f64

CU_LAUNCH_PARAM_END :: rawptr(uintptr(0x00))
CU_LAUNCH_PARAM_BUFFER_POINTER :: rawptr(uintptr(0x01))
CU_LAUNCH_PARAM_BUFFER_SIZE :: rawptr(uintptr(0x02))

CUresult :: enum(c.int) {
    SUCCESS                              = 0,
    ERROR_INVALID_VALUE                  = 1,
    ERROR_OUT_OF_MEMORY                  = 2,
    ERROR_NOT_INITIALIZED                = 3,
    ERROR_DEINITIALIZED                  = 4,
    ERROR_PROFILER_DISABLED              = 5,
    ERROR_PROFILER_NOT_INITIALIZED       = 6,
    ERROR_PROFILER_ALREADY_STARTED       = 7,
    ERROR_PROFILER_ALREADY_STOPPED       = 8,
    ERROR_STUB_LIBRARY                   = 34,
    ERROR_NO_DEVICE                      = 100,
    ERROR_INVALID_DEVICE                 = 101,
    ERROR_DEVICE_NOT_LICENSED            = 102,
    ERROR_INVALID_IMAGE                  = 200,
    ERROR_INVALID_CONTEXT                = 201,
    ERROR_CONTEXT_ALREADY_CURRENT        = 202,
    ERROR_MAP_FAILED                     = 205,
    ERROR_UNMAP_FAILED                   = 206,
    ERROR_ARRAY_IS_MAPPED                = 207,
    ERROR_ALREADY_MAPPED                 = 208,
    ERROR_NO_BINARY_FOR_GPU              = 209,
    ERROR_ALREADY_ACQUIRED               = 210,
    ERROR_NOT_MAPPED                     = 211,
    ERROR_NOT_MAPPED_AS_ARRAY            = 212,
    ERROR_NOT_MAPPED_AS_POINTER          = 213,
    ERROR_ECC_UNCORRECTABLE              = 214,
    ERROR_UNSUPPORTED_LIMIT              = 215,
    ERROR_CONTEXT_ALREADY_IN_USE         = 216,
    ERROR_PEER_ACCESS_UNSUPPORTED        = 217,
    ERROR_INVALID_PTX                    = 218,
    ERROR_INVALID_GRAPHICS_CONTEXT       = 219,
    ERROR_NVLINK_UNCORRECTABLE           = 220,
    ERROR_JIT_COMPILER_NOT_FOUND         = 221,
    ERROR_UNSUPPORTED_PTX_VERSION        = 222,
    ERROR_JIT_COMPILATION_DISABLED       = 223,
    ERROR_UNSUPPORTED_EXEC_AFFINITY      = 224,
    ERROR_INVALID_SOURCE                 = 300,
    ERROR_FILE_NOT_FOUND                 = 301,
    ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND = 302,
    ERROR_SHARED_OBJECT_INIT_FAILED      = 303,
    ERROR_OPERATING_SYSTEM               = 304,
    ERROR_INVALID_HANDLE                 = 400,
    ERROR_ILLEGAL_STATE                  = 401,
    ERROR_NOT_FOUND                      = 500,
    ERROR_NOT_READY                      = 600,
    ERROR_ILLEGAL_ADDRESS                = 700,
    ERROR_LAUNCH_OUT_OF_RESOURCES        = 701,
    ERROR_LAUNCH_TIMEOUT                 = 702,
    ERROR_LAUNCH_INCOMPATIBLE_TEXTURING  = 703,
    ERROR_PEER_ACCESS_ALREADY_ENABLED    = 704,
    ERROR_PEER_ACCESS_NOT_ENABLED        = 705,
    ERROR_PRIMARY_CONTEXT_ACTIVE         = 708,
    ERROR_CONTEXT_IS_DESTROYED           = 709,
    ERROR_ASSERT                         = 710,
    ERROR_TOO_MANY_PEERS                 = 711,
    ERROR_HOST_MEMORY_ALREADY_REGISTERED = 712,
    ERROR_HOST_MEMORY_NOT_REGISTERED     = 713,
    ERROR_HARDWARE_STACK_ERROR           = 714,
    ERROR_ILLEGAL_INSTRUCTION            = 715,
    ERROR_MISALIGNED_ADDRESS             = 716,
    ERROR_INVALID_ADDRESS_SPACE          = 717,
    ERROR_INVALID_PC                     = 718,
    ERROR_LAUNCH_FAILED                  = 719,
    ERROR_COOPERATIVE_LAUNCH_TOO_LARGE   = 720,
    ERROR_NOT_PERMITTED                  = 800,
    ERROR_NOT_SUPPORTED                  = 801,
    ERROR_SYSTEM_NOT_READY               = 802,
    ERROR_SYSTEM_DRIVER_MISMATCH         = 803,
    ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE = 804,
    ERROR_MPS_CONNECTION_FAILED          = 805,
    ERROR_MPS_RPC_FAILURE                = 806,
    ERROR_MPS_SERVER_NOT_READY           = 807,
    ERROR_MPS_MAX_CLIENTS_REACHED        = 808,
    ERROR_MPS_MAX_CONNECTIONS_REACHED    = 809,
    ERROR_STREAM_CAPTURE_UNSUPPORTED     = 900,
    ERROR_STREAM_CAPTURE_INVALIDATED     = 901,
    ERROR_STREAM_CAPTURE_MERGE           = 902,
    ERROR_STREAM_CAPTURE_UNMATCHED       = 903,
    ERROR_STREAM_CAPTURE_UNJOINED        = 904,
    ERROR_STREAM_CAPTURE_ISOLATION       = 905,
    ERROR_STREAM_CAPTURE_IMPLICIT        = 906,
    ERROR_CAPTURED_EVENT                 = 907,
    ERROR_STREAM_CAPTURE_WRONG_THREAD    = 908,
    ERROR_TIMEOUT                        = 909,
    ERROR_GRAPH_EXEC_UPDATE_FAILURE      = 910,
    ERROR_EXTERNAL_DEVICE                = 911,
    ERROR_UNKNOWN                        = 999
}


/**
 * Context creation flags
 */
CUctx_flag :: enum(c.int) {
    /**< Automatic scheduling */
    SCHED_AUTO,
    /**< Set spin as default scheduling */
    SCHED_SPIN,
    /**< Set yield as default scheduling */
    SCHED_YIELD,
    /**< Set blocking synchronization as default scheduling */
    SCHED_BLOCKING_SYNC,
    /**< \deprecated This flag was deprecated as of CUDA 11.0 and it no longer has any effect. All contexts as of CUDA 3.2 behave as though the flag is enabled. */
    MAP_HOST,
    /**< Keep local memory allocation after launch */
    LMEM_RESIZE_TO_MAX,
}
CUctx_flags :: bit_set[CUctx_flag; c.int]

CUstream_flags :: enum(c.int) {
    /**< Default stream flag */
    DEFAULT             = 0x0,
    /**< Stream does not synchronize with stream 0 (the NULL stream) */
    NON_BLOCKING        = 0x1
}

CUjit_option :: enum(c.int) {
    MAX_REGISTERS = 0, THREADS_PER_BLOCK, WALL_TIME, INFO_LOG_BUFFER,
    INFO_LOG_BUFFER_SIZE_BYTES, ERROR_LOG_BUFFER, ERROR_LOG_BUFFER_SIZE_BYTES,
    OPTIMIZATION_LEVEL, TARGET_FROM_CUCONTEXT, TARGET, FALLBACK_STRATEGY,
    GENERATE_DEBUG_INFO, LOG_VERBOSE, GENERATE_LINE_INFO, CACHE_MODE,
    NEW_SM3X_OPT, FAST_COMPILE, GLOBAL_SYMBOL_NAMES, GLOBAL_SYMBOL_ADDRESSES,
    GLOBAL_SYMBOL_COUNT, LTO, FTZ, PREC_DIV, PREC_SQRT, FMA, NUM_OPTIONS,
}

CUmemAttach :: enum {
    /**< Memory can be accessed by any stream on any device */
    CU_MEM_ATTACH_GLOBAL,
    /**< Memory cannot be accessed by any stream on any device */
    CU_MEM_ATTACH_HOST,
    /**< Memory can only be accessed by a single stream on the associated device */
    CU_MEM_ATTACH_SINGLE,
}

CUmem_advise :: enum(c.int) {
    /**< Data will mostly be read and only occasionally be written to */
    SET_READ_MOSTLY          = 1,
    /**< Undo the effect of ::CU_MEM_ADVISE_SET_READ_MOSTLY */
    UNSET_READ_MOSTLY        = 2,
    /**< Set the preferred location for the data as the specified device */
    SET_PREFERRED_LOCATION   = 3,
    /**< Clear the preferred location for the data */
    UNSET_PREFERRED_LOCATION = 4,
    /**< Data will be accessed by the specified device, so prevent page faults as much as possible */
    SET_ACCESSED_BY          = 5,
    /**< Let the Unified Memory subsystem decide on the page faulting policy for the specified device */
    UNSET_ACCESSED_BY        = 6,
}

CUmem_range_attribute :: enum(c.int) {
    /**< Whether the range will mostly be read and only occasionally be written to */
    READ_MOSTLY            = 1,
    /**< The preferred location of the range */
    PREFERRED_LOCATION     = 2,
    /**< Memory range has ::CU_MEM_ADVISE_SET_ACCESSED_BY set for specified device */
    ACCESSED_BY            = 3,
    /**< The last location to which the range was prefetched */
    LAST_PREFETCH_LOCATION = 4,
}

CUDA_LAUNCH_PARAMS :: struct {
    /**< Kernel to launch */
    function: CUfunction,
    /**< Width of grid in blocks */
    gridDimX: c.uint,
    /**< Height of grid in blocks */
    gridDimY: c.uint,
    /**< Depth of grid in blocks */
    gridDimZ: c.uint,
    /**< X dimension of each thread block */
    blockDimX: c.uint,
    /**< Y dimension of each thread block */
    blockDimY: c.uint,
    /**< Z dimension of each thread block */
    blockDimZ: c.uint,
    /**< Dynamic shared-memory size per thread block in bytes */
    sharedMemBytes: c.uint,
    /**< Stream identifier */
    hStream: CUstream,
    /**< Array of pointers to kernel parameters */
    kernelParams: [^]rawptr,
}
CUfunction_attribute :: enum(c.int) {
    /** The maximum number of threads per block, beyond which a launch of the function would fail. */
    MAX_THREADS_PER_BLOCK = 0,
    /** The size in bytes of statically-allocated shared memory required by this function. */
    SHARED_SIZE_BYTES = 1,
    /** The size in bytes of user-allocated constant memory required by this function. */
    CONST_SIZE_BYTES = 2,
    /** The size in bytes of local memory used by each thread of this function. */
    LOCAL_SIZE_BYTES = 3,
    /** The number of registers used by each thread of this function. */
    NUM_REGS = 4,
    /**
     * The PTX virtual architecture version for which the function was
     * compiled. This value is the major PTX version * 10 + the minor PTX
     * version, so a PTX version 1.3 function would return the value 13.
     * Note that this may return the undefined value of 0 for cubins
     * compiled prior to CUDA 3.0.
     */
    PTX_VERSION = 5,
    /**
     * The binary architecture version for which the function was compiled.
     * This value is the major binary version * 10 + the minor binary version,
     * so a binary version 1.3 function would return the value 13. Note that
     * this will return a value of 10 for legacy cubins that do not have a
     * properly-encoded binary architecture version.
     */
    BINARY_VERSION = 6,
    /** The attribute to indicate whether the function has been compiled with user specified option "-Xptxas --dlcm=ca" set . */
    CACHE_MODE_CA = 7,
    /** The maximum size in bytes of dynamically-allocated shared memory that can be used by this function. */
    MAX_DYNAMIC_SHARED_SIZE_BYTES = 8,
    /**
     * On devices where the L1 cache and shared memory use the same hardware resources,
     * this sets the shared memory carveout preference, in percent of the total shared memory.
     * Refer to ::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR.
     * This is only a hint, and the driver can choose a different ratio if required to execute the function.
     * See ::cuFuncSetAttribute
     */
    PREFERRED_SHARED_MEMORY_CARVEOUT = 9,
    MAX
}

CUfunc_cache :: enum(c.int) {
    /**< no preference for shared memory or L1 (default) */
    NONE   = 0,
    /**< prefer larger shared memory and smaller L1 cache */
    SHARED = 1,
    /**< prefer larger L1 cache and smaller shared memory */
    L1     = 2,
    /**< prefer equal sized L1 cache and shared memory */
    EQUAL  = 3,
}

CUsharedconfig :: enum(c.int) {
    /**< set default shared memory bank size */
    DEFAULT_BANK_SIZE    = 0,
    /**< set shared memory bank width to four bytes */
    FOUR_BYTE_BANK_SIZE  = 1,
    /**< set shared memory bank width to eight bytes */
    EIGHT_BYTE_BANK_SIZE = 2,
}

CUDA_KERNEL_NODE_PARAMS :: struct {
    func:           CUfunction,
    gridDimX:       c.uint,
    gridDimY:       c.uint,
    gridDimZ:       c.uint,
    blockDimX:      c.uint,
    blockDimY:      c.uint,
    blockDimZ:      c.uint,
    sharedMemBytes: c.uint,
    kernelParams:   [^]rawptr,
    extra:          [^]rawptr,
}

CUDA_MEMSET_NODE_PARAMS :: struct {
    dst:         CUdeviceptr,
    pitch:       c.size_t,
    value:       c.uint,
    elementSize: c.uint,
    width:       c.size_t,
    height:      c.size_t,
}

CUmemorytype :: enum(c.int) {
    /**< Host memory */
    HOST    = 1,
    /**< Device memory */
    DEVICE  = 2,
    /**< Array memory */
    ARRAY   = 3,
    /**< Unified device or host memory */
    UNIFIED = 4,
}

CUDA_MEMCPY3D :: struct {
    /**< Source X in bytes */
    srcXInBytes:   c.size_t,
    /**< Source Y */
    srcY:          c.size_t,
    /**< Source Z */
    srcZ:          c.size_t,
    /**< Source LOD */
    srcLOD:        c.size_t,
    /**< Source memory type (host, device, array) */
    srcMemoryType: CUmemorytype,
    /**< Source host pointer */
    srcHost:       rawptr,
    /**< Source device pointer */
    srcDevice:     CUdeviceptr,
    /**< Source array reference */
    srcArray:      CUarray,
    /**< Must be NULL */
    reserved0:     rawptr,
    /**< Source pitch (ignored when src is array) */
    srcPitch:      c.size_t,
    /**< Source height (ignored when src is array; may be 0 if Depth==1) */
    srcHeight:     c.size_t,
    /**< Destination X in bytes */
    dstXInBytes:   c.size_t,
    /**< Destination Y */
    dstY:          c.size_t,
    /**< Destination Z */
    dstZ:          c.size_t,
    /**< Destination LOD */
    dstLOD:        c.size_t,
    /**< Destination memory type (host, device, array) */
    dstMemoryType: CUmemorytype,
    /**< Destination host pointer */
    dstHost:       rawptr,
    /**< Destination device pointer */
    dstDevice:     CUdeviceptr,
    /**< Destination array reference */
    dstArray:      CUarray,
    /**< Must be NULL */
    reserved1:     rawptr,
    /**< Destination pitch (ignored when dst is array) */
    dstPitch:      c.size_t,
    /**< Destination height (ignored when dst is array; may be 0 if Depth==1) */
    dstHeight:     c.size_t,
    /**< Width of 3D memory copy in bytes */
    WidthInBytes:  c.size_t,
    /**< Height of 3D memory copy */
    Height:        c.size_t,
    /**< Depth of 3D memory copy */
    Depth:         c.size_t,
}

CUDA_HOST_NODE_PARAMS :: struct {
    fn:       proc "c" (rawptr), /**< The function to call when the node executes */
    userData: rawptr,   /**< Argument to pass to the function */
}

CUgraphMem_attribute :: enum(c.int) {
    /** (value type = cuuint64_t) Amount of memory, in bytes, currently associated with graphs */
    USED_MEM_CURRENT,
    /** (value type = cuuint64_t) High watermark of memory, in bytes, associated with graphs since the last time it was reset.  High watermark can only be reset to zero. */
    USED_MEM_HIGH,
    /** (value type = cuuint64_t) Amount of memory, in bytes, currently allocated for use by the CUDA graphs asynchronous allocator. */
    RESERVED_MEM_CURRENT,
    /** (value type = cuuint64_t) High watermark of memory, in bytes, currently allocated for use by the CUDA graphs asynchronous allocator. */
    RESERVED_MEM_HIGH,
}

CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS :: struct {
    params: struct {
        /** Parameters for fence objects */
        fence: struct {
            /** Value of fence to be signaled */
            value: c.ulonglong,
        },
        nvSciSync: struct #raw_union {
            /** Pointer to NvSciSyncFence. Valid if ::CUexternalSemaphoreHandleType is of type ::CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_NVSCISYNC. */
            fence: rawptr,
            reserved: c.ulonglong,
        },
        /** Parameters for keyed mutex objects */
        keyedMutex: struct {
            /** Value of key to release the mutex with */
            key: c.ulonglong,
        },
        reserved: [12]c.uint,
    },
    /** Only when ::CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS is used to signal a ::CUexternalSemaphore of type ::CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_NVSCISYNC, the valid flag is ::CUDA_EXTERNAL_SEMAPHORE_SIGNAL_SKIP_NVSCIBUF_MEMSYNC which indicates that while signaling the ::CUexternalSemaphore, no memory synchronization operations should be performed for any external memory object imported as ::CU_EXTERNAL_MEMORY_HANDLE_TYPE_NVSCIBUF. For all other types of ::CUexternalSemaphore, flags must be zero. */
    flags: c.uint,
    reserved: [16]c.uint,
}
CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS :: struct {
    params:   struct {
        fence:    struct {
            value: c.ulonglong,
        },
        nvSciSync: struct #raw_union {
            fence:    rawptr,
            reserved: c.ulonglong,
        },
        keyedMutex: struct {
            key: c.ulonglong,
            /** Timeout in milliseconds to wait to acquire the mutex */
            timeoutMs: c.uint,
        },
        reserved: [10]c.uint,
    },
    /** Only when ::CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS is used to wait on a ::CUexternalSemaphore of type ::CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_NVSCISYNC, the valid flag is ::CUDA_EXTERNAL_SEMAPHORE_WAIT_SKIP_NVSCIBUF_MEMSYNC which indicates that while waiting for the ::CUexternalSemaphore, no memory synchronization operations should be performed for any external memory object imported as ::CU_EXTERNAL_MEMORY_HANDLE_TYPE_NVSCIBUF. For all other types of ::CUexternalSemaphore, flags must be zero. */
    flags: c.uint,
    reserved: [16]c.uint,
}

CUDA_EXT_SEM_WAIT_NODE_PARAMS :: struct {
    /**< Array of external semaphore handles. */
    extSemArray: ^CUexternalSemaphore,
    /**< Array of external semaphore wait parameters. */
    paramsArray: ^CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS,
    /**< Number of handles and parameters supplied in extSemArray and paramsArray. */
    numExtSems:  c.uint,
}

CUDA_EXT_SEM_SIGNAL_NODE_PARAMS :: struct {
    /**< Array of external semaphore handles. */
    extSemArray: ^CUexternalSemaphore,
    /**< Array of external semaphore signal parameters. */
    paramsArray: ^CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS,
    /**< Number of handles and parameters supplied in extSemArray and paramsArray. */
    numExtSems:  c.uint,
}

CUgraphNodeType :: enum(c.int) {
    /**< GPU kernel node */
    KERNEL           = 0,
    /**< Memcpy node */
    MEMCPY           = 1,
    /**< Memset node */
    MEMSET           = 2,
    /**< Host (executable) node */
    HOST             = 3,
    /**< Node which executes an embedded graph */
    GRAPH            = 4,
    /**< Empty (no-op) node */
    EMPTY            = 5,
    /**< External event wait node */
    WAIT_EVENT       = 6,
    /**< External event record node */
    EVENT_RECORD     = 7,
    /**< External semaphore signal node */
    EXT_SEMAS_SIGNAL = 8,
    /**< External semaphore wait node */
    EXT_SEMAS_WAIT   = 9,
    /**< Memory Allocation Node */
    MEM_ALLOC        = 10,
    /**< Memory Free Node */
    MEM_FREE         = 11,
}

CUmemLocationType :: enum(c.int) {
    INVALID = 0,
    /**< Location is a device location, thus id is a device ordinal */
    DEVICE  = 1,
    MAX     = max(c.int),
}

CUmemAllocationType :: enum(c.int) {
    INVALID = 0,
    /** This allocation type is 'pinned', i.e. cannot migrate from its current location while the application is actively using it */
    PINNED = 1,
    MAX = max(c.int),
}

CUmemAllocationGranularity_flags :: enum(c.int) {
    /**< Minimum required granularity for allocation */
    MINIMUM     = 0,
    /**< Recommended granularity for allocation for best performance */
    RECOMMENDED = 1,
}

CUmemOperationType :: enum(c.int) {
    MAP   = 1,
    UNMAP = 2,
}

CUmemAllocationHandleType :: enum(c.int) {
    /**< Does not allow any export mechanism. > */
    NONE                  = 0,
    /**< Allows a file descriptor to be used for exporting. Permitted only on POSIX systems. (int) */
    POSIX_FILE_DESCRIPTOR = 1,
    /**< Allows a Win32 NT handle to be used for exporting. (HANDLE) */
    WIN32                 = 2,
    /**< Allows a Win32 KMT handle to be used for exporting. (D3DKMT_HANDLE) */
    WIN32_KMT             = 4,
    MAX                   = max(c.int),
}
CUmemLocation :: struct {
    /**< Specifies the location type, which modifies the meaning of id. */
    type: CUmemLocationType,
    /**< identifier for a given this location's ::CUmemLocationType. */
    id:   c.int,
}
CUmemPoolProps :: struct {
    /**< Allocation type. Currently must be specified as CU_MEM_ALLOCATION_TYPE_PINNED */
    allocType:   CUmemAllocationType,
    /**< Handle types that will be supported by allocations from the pool. */
    handleTypes: CUmemAllocationHandleType,
    /**< Location where allocations should reside. */
    location:    CUmemLocation,

    /**
    * Windows-specific LPSECURITYATTRIBUTES required when
    * ::CU_MEM_HANDLE_TYPE_WIN32 is specified.  This security attribute defines
    * the scope of which exported allocations may be tranferred to other
    * processes.  In all other cases, this field is required to be zero.
    */
    win32SecurityAttributes: rawptr,
    reserved:    [64]c.uchar,               /**< reserved for future use, must be 0 */
}

/**
* Specifies the memory protection flags for mapping.
*/
CUmemAccess_flags :: enum(c.int) {
    NONE      = 0, /**< Default, make the address range not accessible */
    READ      = 1, /**< Make the address range read accessible */
    READWRITE = 3, /**< Make the address range read-write accessible */
    MAX       = max(c.int),
}

CUmemAccessDesc :: struct {
    location: CUmemLocation,     /**< Location on which the request is to change it's accessibility */
    flags:    CUmemAccess_flags, /**< ::CUmemProt accessibility flags to set on the request */
}

CUDA_MEM_ALLOC_NODE_PARAMS :: struct {
    /**
    * in: location where the allocation should reside (specified in ::location).
    * ::handleTypes must be ::CU_MEM_HANDLE_TYPE_NONE. IPC is not supported.
    */
    poolProps: CUmemPoolProps,
    /**< in: array of memory access descriptors. Used to describe peer GPU access */
    accessDescs:     ^CUmemAccessDesc,
    /**< in: number of memory access descriptors.  Must not exceed the number of GPUs. */
    accessDescCount: c.size_t,
    /**< in: size in bytes of the requested allocation */
    bytesize:        c.size_t,
    /**< out: address of the allocation returned by CUDA */
    dptr:            CUdeviceptr,
}

CUgraphExecUpdateResult :: enum(c.int) {
    /**< The update succeeded */
    SUCCESS                           = 0,
    /**< The update failed for an unexpected reason which is described in the return value of the function */
    ERROR                             = 1,
    /**< The update failed because the topology changed */
    ERROR_TOPOLOGY_CHANGED            = 2,
    /**< The update failed because a node type changed */
    ERROR_NODE_TYPE_CHANGED           = 3,
    /**< The update failed because the function of a kernel node changed (CUDA driver < 11.2) */
    ERROR_FUNCTION_CHANGED            = 4,
    /**< The update failed because the parameters changed in a way that is not supported */
    ERROR_PARAMETERS_CHANGED          = 5,
    /**< The update failed because something about the node is not supported */
    ERROR_NOT_SUPPORTED               = 6,
    /**< The update failed because the function of a kernel node changed in an unsupported way */
    ERROR_UNSUPPORTED_FUNCTION_CHANGE = 7,
}

CUkernelNodeAttrID :: enum(c.int) {
    /**< Identifier for ::CUkernelNodeAttrValue::accessPolicyWindow. */
    ACCESS_POLICY_WINDOW = 1,
    /**< Allows a kernel node to be cooperative (see ::cuLaunchCooperativeKernel). */
    COOPERATIVE          = 2,
}

/**
* Specifies performance hint with ::CUaccessPolicyWindow for hitProp and missProp members.
*/
CUaccessProperty :: enum(c.int) {
    /**< Normal cache persistence. */
    NORMAL     = 0,
    /**< Streaming access is less likely to persit from cache. */
    STREAMING  = 1,
    /**< Persisting access is more likely to persist in cache.*/
    PERSISTING = 2,
}

/**
* Specifies an access policy for a window, a contiguous extent of memory
* beginning at base_ptr and ending at base_ptr + num_bytes.
* num_bytes is limited by CU_DEVICE_ATTRIBUTE_MAX_ACCESS_POLICY_WINDOW_SIZE.
* Partition into many segments and assign segments such that:
* sum of "hit segments" / window == approx. ratio.
* sum of "miss segments" / window == approx 1-ratio.
* Segments and ratio specifications are fitted to the capabilities of
* the architecture.
* Accesses in a hit segment apply the hitProp access policy.
* Accesses in a miss segment apply the missProp access policy.
*/
CUaccessPolicyWindow :: struct {
    /**< Starting address of the access policy window. CUDA driver may align it. */
    base_ptr:  rawptr,
    /**< Size in bytes of the window policy. CUDA driver may restrict the maximum size and alignment. */
    num_bytes: c.size_t,
    /**< hitRatio specifies percentage of lines assigned hitProp, rest are assigned missProp. */
    hitRatio:  f32,
    /**< ::CUaccessProperty set for hit. */
    hitProp:   CUaccessProperty,
    /**< ::CUaccessProperty set for miss. Must be either NORMAL or STREAMING */
    missProp:  CUaccessProperty,
}


/**
* Graph kernel node attributes union, used with ::cuKernelNodeSetAttribute/::cuKernelNodeGetAttribute
*/
CUkernelNodeAttrValue :: struct #raw_union {
    /**< Attribute ::CUaccessPolicyWindow. */
    accessPolicyWindow: CUaccessPolicyWindow,
    /**< Nonzero indicates a cooperative kernel (see ::cuLaunchCooperativeKernel). */
    cooperative:        c.int,
}



/**
* Possible modes for stream capture thread interactions. For more details see
* ::cuStreamBeginCapture and ::cuThreadExchangeStreamCaptureMode
*/
CUstreamCaptureMode :: enum(c.int) {
    GLOBAL       = 0,
    THREAD_LOCAL = 1,
    RELAXED      = 2,
}

/**
* Stream Attributes
*/
CUstreamAttrID :: enum(c.int) {
    ACCESS_POLICY_WINDOW   = 1, /**< Identifier for ::CUstreamAttrValue::accessPolicyWindow. */
    SYNCHRONIZATION_POLICY = 3, /**< ::CUsynchronizationPolicy for work queued up in this stream */
}

/**
* Stream attributes union, used with ::cuStreamSetAttribute/::cuStreamGetAttribute
*/
CUstreamAttrValue :: struct #raw_union {
    accessPolicyWindow: CUaccessPolicyWindow,    /**< Attribute ::CUaccessPolicyWindow. */
    syncPolicy:         CUsynchronizationPolicy, /**< Value for ::CU_STREAM_ATTRIBUTE_SYNCHRONIZATION_POLICY. */
}

/**
* Possible stream capture statuses returned by ::cuStreamIsCapturing
*/
CUstreamCaptureStatus :: enum(c.int) {
    NONE        = 0, /**< Stream is not capturing */
    ACTIVE      = 1, /**< Stream is actively capturing */
    INVALIDATED = 2, /**< Stream is part of a capture sequence that
                                                   has been invalidated, but not terminated */
}

CUsynchronizationPolicy :: enum(c.int) {
    CU_SYNC_POLICY_AUTO = 1,
    CU_SYNC_POLICY_SPIN = 2,
    CU_SYNC_POLICY_YIELD = 3,
    CU_SYNC_POLICY_BLOCKING_SYNC = 4
}

CU_MEMHOSTREGISTER :: enum(c.int) {
 /**
  * If set, host memory is portable between CUDA contexts.
  * Flag for ::cuMemHostRegister()
  */
  PORTABLE,

  /**
   * If set, host memory is mapped into CUDA address space and
   * ::cuMemHostGetDevicePointer() may be called on the host pointer.
   * Flag for ::cuMemHostRegister()
   */
  DEVICEMAP,

  /**
   * If set, the passed memory pointer is treated as pointing to some
   * memory-mapped I/O space, e.g. belonging to a third-party PCIe device.
   * On Windows the flag is a no-op.
   * On Linux that memory is marked as non cache-coherent for the GPU and
   * is expected to be physically contiguous. It may return
   * ::CUDA_ERROR_NOT_PERMITTED if run as an unprivileged user,
   * ::CUDA_ERROR_NOT_SUPPORTED on older Linux kernel versions.
   * On all other platforms, it is not supported and ::CUDA_ERROR_NOT_SUPPORTED
   * is returned.
   * Flag for ::cuMemHostRegister()
   */
  IOMEMORY,

  /**
  * If set, the passed memory pointer is treated as pointing to memory that is
  * considered read-only by the device.  On platforms without
  * ::CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES, this flag is
  * required in order to register memory mapped to the CPU as read-only.  Support
  * for the use of this flag can be queried from the device attribute
  * ::CU_DEVICE_ATTRIBUTE_READ_ONLY_HOST_REGISTER_SUPPORTED.  Using this flag with
  * a current context associated with a device that does not have this attribute
  * set will cause ::cuMemHostRegister to error with ::CUDA_ERROR_NOT_SUPPORTED.
  */
  READ_ONLY,
}

CUgraphicsMapResourceFlags :: enum(c.int) {
    CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE          = 0x00,
    CU_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY     = 0x01,
    CU_GRAPHICS_MAP_RESOURCE_FLAGS_WRITE_DISCARD = 0x02
}

CUipcMem_flags :: bit_set[enum c.int {
    LAZY_ENABLE_PEER_ACCESS /**< Automatically enable peer access between remote devices as needed */
}; c.int]

CUarray_format :: enum(c.int) {
    /**< Unsigned 8-bit integers */
    UNSIGNED_INT8  = 1,
    /**< Unsigned 16-bit integers */
    UNSIGNED_INT16 = 2,
    /**< Unsigned 32-bit integers */
    UNSIGNED_INT32 = 3,
    /**< Signed 8-bit integers */
    SIGNED_INT8    = 8,
    /**< Signed 16-bit integers */
    SIGNED_INT16   = 9,
    /**< Signed 32-bit integers */
    SIGNED_INT32   = 10,
    /**< 16-bit floating point */
    HALF           = 16,
    /**< 32-bit floating point */
    FLOAT          = 32,
    /**< 8-bit YUV planar format, with 4:2:0 sampling */
    NV12           = 176,
    /**< 1 channel unsigned 8-bit normalized integer */
    UNORM_INT8X1   = 192,
    /**< 2 channel unsigned 8-bit normalized integer */
    UNORM_INT8X2   = 193,
    /**< 4 channel unsigned 8-bit normalized integer */
    UNORM_INT8X4   = 194,
    /**< 1 channel unsigned 16-bit normalized integer */
    UNORM_INT16X1  = 195,
    /**< 2 channel unsigned 16-bit normalized integer */
    UNORM_INT16X2  = 196,
    /**< 4 channel unsigned 16-bit normalized integer */
    UNORM_INT16X4  = 197,
    /**< 1 channel signed 8-bit normalized integer */
    SNORM_INT8X1   = 198,
    /**< 2 channel signed 8-bit normalized integer */
    SNORM_INT8X2   = 199,
    /**< 4 channel signed 8-bit normalized integer */
    SNORM_INT8X4   = 200,
    /**< 1 channel signed 16-bit normalized integer */
    SNORM_INT16X1  = 201,
    /**< 2 channel signed 16-bit normalized integer */
    SNORM_INT16X2  = 202,
    /**< 4 channel signed 16-bit normalized integer */
    SNORM_INT16X4  = 203,
    /**< 4 channel unsigned normalized block-compressed (BC1 compression) format */
    BC1_UNORM      = 145,
    /**< 4 channel unsigned normalized block-compressed (BC1 compression) format with sRGB encoding*/
    BC1_UNORM_SRGB = 146,
    /**< 4 channel unsigned normalized block-compressed (BC2 compression) format */
    BC2_UNORM      = 147,
    /**< 4 channel unsigned normalized block-compressed (BC2 compression) format with sRGB encoding*/
    BC2_UNORM_SRGB = 148,
    /**< 4 channel unsigned normalized block-compressed (BC3 compression) format */
    BC3_UNORM      = 149,
    /**< 4 channel unsigned normalized block-compressed (BC3 compression) format with sRGB encoding*/
    BC3_UNORM_SRGB = 150,
    /**< 1 channel unsigned normalized block-compressed (BC4 compression) format */
    BC4_UNORM      = 151,
    /**< 1 channel signed normalized block-compressed (BC4 compression) format */
    BC4_SNORM      = 152,
    /**< 2 channel unsigned normalized block-compressed (BC5 compression) format */
    BC5_UNORM      = 153,
    /**< 2 channel signed normalized block-compressed (BC5 compression) format */
    BC5_SNORM      = 154,
    /**< 3 channel unsigned half-float block-compressed (BC6H compression) format */
    BC6H_UF16      = 155,
    /**< 3 channel signed half-float block-compressed (BC6H compression) format */
    BC6H_SF16      = 156,
    /**< 4 channel unsigned normalized block-compressed (BC7 compression) format */
    BC7_UNORM      = 157,
    /**< 4 channel unsigned normalized block-compressed (BC7 compression) format with sRGB encoding */
    BC7_UNORM_SRGB = 158,
}

CUDA_ARRAY_DESCRIPTOR :: struct {
    /**< Width of array */
    Width:       c.size_t,
    /**< Height of array */
    Height:      c.size_t,
    /**< Array format */
    Format:      CUarray_format,
    /**< Channels per array element */
    NumChannels: c.uint,
}

CUDA_ARRAY3D_DESCRIPTOR :: struct {
    /**< Width of 3D array */
    Width: c.size_t,
    /**< Height of 3D array */
    Height: c.size_t,
    /**< Depth of 3D array */
    Depth: c.size_t,
    /**< Array format */
    Format: CUarray_format,
    /**< Channels per array element */
    NumChannels: c.uint,
    /**< Flags */
    Flags: c.uint,
}

//cuda is a big fan of opaque pointers
CUdevice :: distinct i32
CUcontext :: distinct rawptr
CUmodule :: distinct rawptr
CUfunction :: distinct rawptr
CUarray :: distinct rawptr
CUmipmappedArray :: distinct rawptr
CUtexref :: distinct rawptr
CUsurfref :: distinct rawptr
CUevent :: distinct rawptr
CUstream :: distinct rawptr
CUdeviceptr :: distinct uintptr
CUexternalMemory :: distinct rawptr
CUexternalSemaphore :: distinct rawptr
CUgraph :: distinct rawptr
CUgraphNode :: distinct rawptr
CUgraphExec :: distinct rawptr
CUmemoryPool :: distinct rawptr
CUuserObject :: distinct rawptr
CUgraphicsResource :: distinct rawptr
CUlinkState :: distinct rawptr
CUtexObject :: distinct c.ulonglong
CUsurfObject :: distinct c.ulonglong

CUmemGenericAllocationHandle :: c.ulonglong
CU_DEVICE_CPU:CUdevice: -1

CUjitInputType :: enum(c.int)
{
    /** Compiled device-class-specific device code. Applicable options: none */
    CU_JIT_INPUT_CUBIN = 0,
    /** PTX source code. Applicable options: PTX compiler options */
    CU_JIT_INPUT_PTX,
    /** Bundle of multiple cubins and/or PTX of some device code. Applicable options: PTX compiler options, ::CU_JIT_FALLBACK_STRATEGY */
    CU_JIT_INPUT_FATBINARY,
    /** Host object with embedded device code. Applicable options: PTX compiler options, ::CU_JIT_FALLBACK_STRATEGY */
    CU_JIT_INPUT_OBJECT,
    /** Archive of host objects with embedded device code. Applicable options: PTX compiler options, ::CU_JIT_FALLBACK_STRATEGY */
    CU_JIT_INPUT_LIBRARY,
    /** High-level intermediate code for link-time optimization. Applicable options: NVVM compiler options, PTX compiler options */
    CU_JIT_INPUT_NVVM,
    CU_JIT_NUM_INPUT_TYPES
}


CUuuid :: struct {
    bytes: [16]c.char,
}

CUipcEventHandle :: struct {
    reserved: [64]c.char,
}

CUipcMemHandle :: struct {
    reserved: [64]c.char,
}

CUDA_MEMCPY2D :: struct {
    /**< Source X in bytes */
    srcXInBytes:   c.size_t,
    /**< Source Y */
    srcY:          c.size_t,
    /**< Source memory type (host, device, array) */
    srcMemoryType: CUmemorytype,
    /**< Source host pointer */
    srcHost:       rawptr,
    /**< Source device pointer */
    srcDevice:     CUdeviceptr,
    /**< Source array reference */
    srcArray:      CUarray,
    /**< Source pitch (ignored when src is array) */
    srcPitch:      c.size_t,
    /**< Destination X in bytes */
    dstXInBytes:   c.size_t,
    /**< Destination Y */
    dstY:          c.size_t,
    /**< Destination memory type (host, device, array) */
    dstMemoryType: CUmemorytype,
    /**< Destination host pointer */
    dstHost:       rawptr,
    /**< Destination device pointer */
    dstDevice:     CUdeviceptr,
    /**< Destination array reference */
    dstArray:      CUarray,
    /**< Destination pitch (ignored when dst is array) */
    dstPitch:      c.size_t,
    /**< Width of 2D memory copy in bytes */
    WidthInBytes:  c.size_t,
    /**< Height of 2D memory copy */
    Height:        c.size_t,
}

CUDA_MEMCPY3D_PEER :: struct {
    /**< Source X in bytes */
    srcXInBytes:   c.size_t,
    /**< Source Y */
    srcY:          c.size_t,
    /**< Source Z */
    srcZ:          c.size_t,
    /**< Source LOD */
    srcLOD:        c.size_t,
    /**< Source memory type (host, device, array) */
    srcMemoryType: CUmemorytype,
    /**< Source host pointer */
    srcHost:       rawptr,
    /**< Source device pointer */
    srcDevice:     CUdeviceptr,
    /**< Source array reference */
    srcArray:      CUarray,
    /**< Source context (ignored with srcMemoryType is ::CU_MEMORYTYPE_ARRAY) */
    srcContext:    CUcontext,
    /**< Source pitch (ignored when src is array) */
    srcPitch:      c.size_t,
    /**< Source height (ignored when src is array; may be 0 if Depth==1) */
    srcHeight:     c.size_t,
    /**< Destination X in bytes */
    dstXInBytes:   c.size_t,
    /**< Destination Y */
    dstY:          c.size_t,
    /**< Destination Z */
    dstZ:          c.size_t,
    /**< Destination LOD */
    dstLOD:        c.size_t,
    /**< Destination memory type (host, device, array) */
    dstMemoryType: CUmemorytype,
    /**< Destination host pointer */
    dstHost:       rawptr,
    /**< Destination device pointer */
    dstDevice:     CUdeviceptr,
    /**< Destination array reference */
    dstArray:      CUarray,
    /**< Destination context (ignored with dstMemoryType is ::CU_MEMORYTYPE_ARRAY) */
    dstContext:    CUcontext,
    /**< Destination pitch (ignored when dst is array) */
    dstPitch:      c.size_t,
    /**< Destination height (ignored when dst is array; may be 0 if Depth==1) */
    dstHeight:     c.size_t,
    /**< Width of 3D memory copy in bytes */
    WidthInBytes:  c.size_t,
    /**< Height of 3D memory copy */
    Height:        c.size_t,
    /**< Depth of 3D memory copy */
    Depth:         c.size_t,
}

CUlimit :: enum(c.int) {
    /**< GPU thread stack size */
    STACK_SIZE                       = 0x00,
    /**< GPU printf FIFO size */
    PRINTF_FIFO_SIZE                 = 0x01,
    /**< GPU malloc heap size */
    MALLOC_HEAP_SIZE                 = 0x02,
    /**< GPU device runtime launch synchronize depth */
    DEV_RUNTIME_SYNC_DEPTH           = 0x03,
    /**< GPU device runtime pending launch count */
    DEV_RUNTIME_PENDING_LAUNCH_COUNT = 0x04,
    /**< A value between 0 and 128 that indicates the maximum fetch granularity of L2 (in Bytes). This is a hint */
    MAX_L2_FETCH_GRANULARITY         = 0x05,
    /**< A size in bytes for L2 persisting lines cache size */
    PERSISTING_L2_CACHE_SIZE         = 0x06,
    MAX
}

CUmemAllocationProp :: struct {
    /** Allocation type */
    type:                 CUmemAllocationType,
    /** requested ::CUmemAllocationHandleType */
    requestedHandleTypes: CUmemAllocationHandleType,
    /** Location of allocation */
    location:             CUmemLocation,
    win32HandleMetaData:  rawptr,
    allocFlags:           struct {
         /**Allocation hint for requesting compressible memory. On devices that support Compute Data Compression, compressible memory can be used to accelerate accesses to data with unstructured sparsity and other compressible data patterns. Applications are expected to query allocation property of the handle obtained with ::cuMemCreate using ::cuMemGetAllocationPropertiesFromHandle to validate if the obtained allocation is compressible or not. Note that compressed memory may not be mappable on all devices. */
        compressionType:      CUmemAllocationCompType,
        gpuDirectRDMACapable: c.uchar,
        usage:                c.ushort,
        reserved:             [4]c.uchar,
    },
}

CUdevice_attribute :: enum(c.int) {
    /**< Maximum number of threads per block */
    MAX_THREADS_PER_BLOCK = 1,
    /**< Maximum block dimension X */
    MAX_BLOCK_DIM_X = 2,
    /**< Maximum block dimension Y */
    MAX_BLOCK_DIM_Y = 3,
    /**< Maximum block dimension Z */
    MAX_BLOCK_DIM_Z = 4,
    /**< Maximum grid dimension X */
    MAX_GRID_DIM_X = 5,
    /**< Maximum grid dimension Y */
    MAX_GRID_DIM_Y = 6,
    /**< Maximum grid dimension Z */
    MAX_GRID_DIM_Z = 7,
    /**< Maximum shared memory available per block in bytes */
    MAX_SHARED_MEMORY_PER_BLOCK = 8,
    /**< Deprecated, use MAX_SHARED_MEMORY_PER_BLOCK */
    SHARED_MEMORY_PER_BLOCK = 8,
    /**< Memory available on device for __constant__ variables in a CUDA C kernel in bytes */
    TOTAL_CONSTANT_MEMORY = 9,
    /**< Warp size in threads */
    WARP_SIZE = 10,
    /**< Maximum pitch in bytes allowed by memory copies */
    MAX_PITCH = 11,
    /**< Maximum number of 32-bit registers available per block */
    MAX_REGISTERS_PER_BLOCK = 12,
    /**< Deprecated, use MAX_REGISTERS_PER_BLOCK */
    REGISTERS_PER_BLOCK = 12,
    /**< Typical clock frequency in kilohertz */
    CLOCK_RATE = 13,
    /**< Alignment requirement for textures */
    TEXTURE_ALIGNMENT = 14,
    /**< Device can possibly copy memory and execute a kernel concurrently. Deprecated. Use instead ASYNC_ENGINE_COUNT. */
    GPU_OVERLAP = 15,
    /**< Number of multiprocessors on device */
    MULTIPROCESSOR_COUNT = 16,
    /**< Specifies whether there is a run time limit on kernels */
    KERNEL_EXEC_TIMEOUT = 17,
    /**< Device is integrated with host memory */
    INTEGRATED = 18,
    /**< Device can map host memory into CUDA address space */
    CAN_MAP_HOST_MEMORY = 19,
    /**< Compute mode (See ::CUcomputemode for details) */
    COMPUTE_MODE = 20,
    /**< Maximum 1D texture width */
    MAXIMUM_TEXTURE1D_WIDTH = 21,
    /**< Maximum 2D texture width */
    MAXIMUM_TEXTURE2D_WIDTH = 22,
    /**< Maximum 2D texture height */
    MAXIMUM_TEXTURE2D_HEIGHT = 23,
    /**< Maximum 3D texture width */
    MAXIMUM_TEXTURE3D_WIDTH = 24,
    /**< Maximum 3D texture height */
    MAXIMUM_TEXTURE3D_HEIGHT = 25,
    /**< Maximum 3D texture depth */
    MAXIMUM_TEXTURE3D_DEPTH = 26,
    /**< Maximum 2D layered texture width */
    MAXIMUM_TEXTURE2D_LAYERED_WIDTH = 27,
    /**< Maximum 2D layered texture height */
    MAXIMUM_TEXTURE2D_LAYERED_HEIGHT = 28,
    /**< Maximum layers in a 2D layered texture */
    MAXIMUM_TEXTURE2D_LAYERED_LAYERS = 29,
    /**< Deprecated, use MAXIMUM_TEXTURE2D_LAYERED_WIDTH */
    MAXIMUM_TEXTURE2D_ARRAY_WIDTH = 27,
    /**< Deprecated, use MAXIMUM_TEXTURE2D_LAYERED_HEIGHT */
    MAXIMUM_TEXTURE2D_ARRAY_HEIGHT = 28,
    /**< Deprecated, use MAXIMUM_TEXTURE2D_LAYERED_LAYERS */
    MAXIMUM_TEXTURE2D_ARRAY_NUMSLICES = 29,
    /**< Alignment requirement for surfaces */
    SURFACE_ALIGNMENT = 30,
    /**< Device can possibly execute multiple kernels concurrently */
    CONCURRENT_KERNELS = 31,
    /**< Device has ECC support enabled */
    ECC_ENABLED = 32,
    /**< PCI bus ID of the device */
    PCI_BUS_ID = 33,
    /**< PCI device ID of the device */
    PCI_DEVICE_ID = 34,
    /**< Device is using TCC driver model */
    TCC_DRIVER = 35,
    /**< Peak memory clock frequency in kilohertz */
    MEMORY_CLOCK_RATE = 36,
    /**< Global memory bus width in bits */
    GLOBAL_MEMORY_BUS_WIDTH = 37,
    /**< Size of L2 cache in bytes */
    L2_CACHE_SIZE = 38,
    /**< Maximum resident threads per multiprocessor */
    MAX_THREADS_PER_MULTIPROCESSOR = 39,
    /**< Number of asynchronous engines */
    ASYNC_ENGINE_COUNT = 40,
    /**< Device shares a unified address space with the host */
    UNIFIED_ADDRESSING = 41,
    /**< Maximum 1D layered texture width */
    MAXIMUM_TEXTURE1D_LAYERED_WIDTH = 42,
    /**< Maximum layers in a 1D layered texture */
    MAXIMUM_TEXTURE1D_LAYERED_LAYERS = 43,
    /**< Deprecated, do not use. */
    CAN_TEX2D_GATHER = 44,
    /**< Maximum 2D texture width if CUDA_ARRAY3D_TEXTURE_GATHER is set */
    MAXIMUM_TEXTURE2D_GATHER_WIDTH = 45,
    /**< Maximum 2D texture height if CUDA_ARRAY3D_TEXTURE_GATHER is set */
    MAXIMUM_TEXTURE2D_GATHER_HEIGHT = 46,
    /**< Alternate maximum 3D texture width */
    MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE = 47,
    /**< Alternate maximum 3D texture height */
    MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE = 48,
    /**< Alternate maximum 3D texture depth */
    MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE = 49,
    /**< PCI domain ID of the device */
    PCI_DOMAIN_ID = 50,
    /**< Pitch alignment requirement for textures */
    TEXTURE_PITCH_ALIGNMENT = 51,
    /**< Maximum cubemap texture width/height */
    MAXIMUM_TEXTURECUBEMAP_WIDTH = 52,
    /**< Maximum cubemap layered texture width/height */
    MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH = 53,
    /**< Maximum layers in a cubemap layered texture */
    MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS = 54,
    /**< Maximum 1D surface width */
    MAXIMUM_SURFACE1D_WIDTH = 55,
    /**< Maximum 2D surface width */
    MAXIMUM_SURFACE2D_WIDTH = 56,
    /**< Maximum 2D surface height */
    MAXIMUM_SURFACE2D_HEIGHT = 57,
    /**< Maximum 3D surface width */
    MAXIMUM_SURFACE3D_WIDTH = 58,
    /**< Maximum 3D surface height */
    MAXIMUM_SURFACE3D_HEIGHT = 59,
    /**< Maximum 3D surface depth */
    MAXIMUM_SURFACE3D_DEPTH = 60,
    /**< Maximum 1D layered surface width */
    MAXIMUM_SURFACE1D_LAYERED_WIDTH = 61,
    /**< Maximum layers in a 1D layered surface */
    MAXIMUM_SURFACE1D_LAYERED_LAYERS = 62,
    /**< Maximum 2D layered surface width */
    MAXIMUM_SURFACE2D_LAYERED_WIDTH = 63,
    /**< Maximum 2D layered surface height */
    MAXIMUM_SURFACE2D_LAYERED_HEIGHT = 64,
    /**< Maximum layers in a 2D layered surface */
    MAXIMUM_SURFACE2D_LAYERED_LAYERS = 65,
    /**< Maximum cubemap surface width */
    MAXIMUM_SURFACECUBEMAP_WIDTH = 66,
    /**< Maximum cubemap layered surface width */
    MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH = 67,
    /**< Maximum layers in a cubemap layered surface */
    MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS = 68,
    /**< Deprecated, do not use. Use cudaDeviceGetTexture1DLinearMaxWidth() or cuDeviceGetTexture1DLinearMaxWidth() instead. */
    MAXIMUM_TEXTURE1D_LINEAR_WIDTH = 69,
    /**< Maximum 2D linear texture width */
    MAXIMUM_TEXTURE2D_LINEAR_WIDTH = 70,
    /**< Maximum 2D linear texture height */
    MAXIMUM_TEXTURE2D_LINEAR_HEIGHT = 71,
    /**< Maximum 2D linear texture pitch in bytes */
    MAXIMUM_TEXTURE2D_LINEAR_PITCH = 72,
    /**< Maximum mipmapped 2D texture width */
    MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH = 73,
    /**< Maximum mipmapped 2D texture height */
    MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT = 74,
    /**< Major compute capability version number */
    COMPUTE_CAPABILITY_MAJOR = 75,
    /**< Minor compute capability version number */
    COMPUTE_CAPABILITY_MINOR = 76,
    /**< Maximum mipmapped 1D texture width */
    MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH = 77,
    /**< Device supports stream priorities */
    STREAM_PRIORITIES_SUPPORTED = 78,
    /**< Device supports caching globals in L1 */
    GLOBAL_L1_CACHE_SUPPORTED = 79,
    /**< Device supports caching locals in L1 */
    LOCAL_L1_CACHE_SUPPORTED = 80,
    /**< Maximum shared memory available per multiprocessor in bytes */
    MAX_SHARED_MEMORY_PER_MULTIPROCESSOR = 81,
    /**< Maximum number of 32-bit registers available per multiprocessor */
    MAX_REGISTERS_PER_MULTIPROCESSOR = 82,
    /**< Device can allocate managed memory on this system */
    MANAGED_MEMORY = 83,
    /**< Device is on a multi-GPU board */
    MULTI_GPU_BOARD = 84,
    /**< Unique id for a group of devices on the same multi-GPU board */
    MULTI_GPU_BOARD_GROUP_ID = 85,
    /**< Link between the device and the host supports native atomic operations (this is a placeholder attribute, and is not supported on any current hardware)*/
    HOST_NATIVE_ATOMIC_SUPPORTED = 86,
    /**< Ratio of single precision performance (in floating-point operations per second) to double precision performance */
    SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO = 87,
    /**< Device supports coherently accessing pageable memory without calling cudaHostRegister on it */
    PAGEABLE_MEMORY_ACCESS = 88,
    /**< Device can coherently access managed memory concurrently with the CPU */
    CONCURRENT_MANAGED_ACCESS = 89,
    /**< Device supports compute preemption. */
    COMPUTE_PREEMPTION_SUPPORTED = 90,
    /**< Device can access host registered memory at the same virtual address as the CPU */
    CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM = 91,
    /**< ::cuStreamBatchMemOp and related APIs are supported. */
    CAN_USE_STREAM_MEM_OPS = 92,
    /**< 64-bit operations are supported in ::cuStreamBatchMemOp and related APIs. */
    CAN_USE_64_BIT_STREAM_MEM_OPS = 93,
    /**< ::CU_STREAM_WAIT_VALUE_NOR is supported. */
    CAN_USE_STREAM_WAIT_VALUE_NOR = 94,
    /**< Device supports launching cooperative kernels via ::cuLaunchCooperativeKernel */
    COOPERATIVE_LAUNCH = 95,
    /**< Deprecated, ::cuLaunchCooperativeKernelMultiDevice is deprecated. */
    COOPERATIVE_MULTI_DEVICE_LAUNCH = 96,
    /**< Maximum optin shared memory per block */
    MAX_SHARED_MEMORY_PER_BLOCK_OPTIN = 97,
    /**< The ::CU_STREAM_WAIT_VALUE_FLUSH flag and the ::CU_STREAM_MEM_OP_FLUSH_REMOTE_WRITES MemOp are supported on the device. See \ref CUDA_MEMOP for additional details. */
    CAN_FLUSH_REMOTE_WRITES = 98,
    /**< Device supports host memory registration via ::cudaHostRegister. */
    HOST_REGISTER_SUPPORTED = 99,
    /**< Device accesses pageable memory via the host's page tables. */
    PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES = 100,
    /**< The host can directly access managed memory on the device without migration. */
    DIRECT_MANAGED_MEM_ACCESS_FROM_HOST = 101,
    /**< Deprecated, Use VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED*/
    VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED = 102,
    /**< Device supports virtual memory management APIs like ::cuMemAddressReserve, ::cuMemCreate, ::cuMemMap and related APIs */
    VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED = 102,
    /**< Device supports exporting memory to a posix file descriptor with ::cuMemExportToShareableHandle, if requested via ::cuMemCreate */
    HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED = 103,
    /**< Device supports exporting memory to a Win32 NT handle with ::cuMemExportToShareableHandle, if requested via ::cuMemCreate */
    HANDLE_TYPE_WIN32_HANDLE_SUPPORTED = 104,
    /**< Device supports exporting memory to a Win32 KMT handle with ::cuMemExportToShareableHandle, if requested via ::cuMemCreate */
    HANDLE_TYPE_WIN32_KMT_HANDLE_SUPPORTED = 105,
    /**< Maximum number of blocks per multiprocessor */
    MAX_BLOCKS_PER_MULTIPROCESSOR = 106,
    /**< Device supports compression of memory */
    GENERIC_COMPRESSION_SUPPORTED = 107,
    /**< Maximum L2 persisting lines capacity setting in bytes. */
    MAX_PERSISTING_L2_CACHE_SIZE = 108,
    /**< Maximum value of CUaccessPolicyWindow::num_bytes. */
    MAX_ACCESS_POLICY_WINDOW_SIZE = 109,
    /**< Device supports specifying the GPUDirect RDMA flag with ::cuMemCreate */
    GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED = 110,
    /**< Shared memory reserved by CUDA driver per block in bytes */
    RESERVED_SHARED_MEMORY_PER_BLOCK = 111,
    /**< Device supports sparse CUDA arrays and sparse CUDA mipmapped arrays */
    SPARSE_CUDA_ARRAY_SUPPORTED = 112,
    /**< Device supports using the ::cuMemHostRegister flag ::CU_MEMHOSTERGISTER_READ_ONLY to register memory that must be mapped as read-only to the GPU */
    READ_ONLY_HOST_REGISTER_SUPPORTED = 113,
    /**< External timeline semaphore interop is supported on the device */
    TIMELINE_SEMAPHORE_INTEROP_SUPPORTED = 114,
    /**< Device supports using the ::cuMemAllocAsync and ::cuMemPool family of APIs */
    MEMORY_POOLS_SUPPORTED = 115,
    /**< Device supports GPUDirect RDMA APIs, like nvidia_p2p_get_pages (see https://docs.nvidia.com/cuda/gpudirect-rdma for more information) */
    GPU_DIRECT_RDMA_SUPPORTED = 116,
    /**< The returned attribute shall be interpreted as a bitmask, where the individual bits are described by the ::CUflushGPUDirectRDMAWritesOptions enum */
    GPU_DIRECT_RDMA_FLUSH_WRITES_OPTIONS = 117,
    /**< GPUDirect RDMA writes to the device do not need to be flushed for consumers within the scope indicated by the returned attribute. See ::CUGPUDirectRDMAWritesOrdering for the numerical values returned here. */
    GPU_DIRECT_RDMA_WRITES_ORDERING = 118,
    /**< Handle types supported with mempool based IPC */
    MEMPOOL_SUPPORTED_HANDLE_TYPES = 119,
    MAX
}

CUpointer_attribute :: enum(c.int) {
    /**< The ::CUcontext on which a pointer was allocated or registered */
    CONTEXT = 1,
    /**< The ::CUmemorytype describing the physical location of a pointer */
    MEMORY_TYPE = 2,
    /**< The address at which a pointer's memory may be accessed on the device */
    DEVICE_POINTER = 3,
    /**< The address at which a pointer's memory may be accessed on the host */
    HOST_POINTER = 4,
    /**< A pair of tokens for use with the nv-p2p.h Linux kernel interface */
    P2P_TOKENS = 5,
    /**< Synchronize every synchronous memory operation initiated on this region */
    SYNC_MEMOPS = 6,
    /**< A process-wide unique ID for an allocated memory region*/
    BUFFER_ID = 7,
    /**< Indicates if the pointer points to managed memory */
    IS_MANAGED = 8,
    /**< A device ordinal of a device on which a pointer was allocated or registered */
    DEVICE_ORDINAL = 9,
    /**< 1 if this pointer maps to an allocation that is suitable for ::cudaIpcGetMemHandle, 0 otherwise **/
    IS_LEGACY_CUDA_IPC_CAPABLE = 10,
    /**< Starting address for this requested pointer */
    RANGE_START_ADDR = 11,
    /**< Size of the address range for this requested pointer */
    RANGE_SIZE = 12,
    /**< 1 if this pointer is in a valid address range that is mapped to a backing allocation, 0 otherwise **/
    MAPPED = 13,
    /**< Bitmask of allowed ::CUmemAllocationHandleType for this allocation **/
    ALLOWED_HANDLE_TYPES = 14,
    /**< 1 if the memory this pointer is referencing can be used with the GPUDirect RDMA API **/
    IS_GPU_DIRECT_RDMA_CAPABLE = 15,
    /**< Returns the access flags the device associated with the current context has on the corresponding memory referenced by the pointer given */
    ACCESS_FLAGS = 16,
    /**< Returns the mempool handle for the allocation if it was allocated from a mempool. Otherwise returns NULL. **/
    MEMPOOL_HANDLE = 17
}

CUmemPool_attribute :: enum(c.int) {
    /** (value type = int) Allow cuMemAllocAsync to use memory asynchronously freed in another streams as long as a stream ordering dependency of the allocating stream on the free action exists. Cuda events and null stream interactions can create the required stream ordered dependencies. (default enabled) */
    REUSE_FOLLOW_EVENT_DEPENDENCIES = 1,
    /** (value type = int) Allow reuse of already completed frees when there is no dependency between the free and allocation. (default enabled) */
    REUSE_ALLOW_OPPORTUNISTIC,
    /** (value type = int) Allow cuMemAllocAsync to insert new stream dependencies in order to establish the stream ordering required to reuse a piece of memory released by cuFreeAsync (default enabled). */
    REUSE_ALLOW_INTERNAL_DEPENDENCIES,
    /** (value type = cuuint64_t) Amount of reserved memory in bytes to hold onto before trying to release memory back to the OS. When more than the release threshold bytes of memory are held by the memory pool, the allocator will try to release memory back to the OS on the next call to stream, event or context synchronize. (default 0) */
    RELEASE_THRESHOLD,
    /** (value type = cuuint64_t) Amount of backing memory currently allocated for the mempool. */
    RESERVED_MEM_CURRENT,
    /** (value type = cuuint64_t) High watermark of backing memory allocated for the mempool since the last time it was reset. High watermark can only be reset to zero. */
    RESERVED_MEM_HIGH,
    /** (value type = cuuint64_t) Amount of memory from the pool that is currently in use by the application. */
    USED_MEM_CURRENT,
    /** (value type = cuuint64_t) High watermark of the amount of memory from the pool that was in use by the application since the last time it was reset. High watermark can only be reset to zero. */
    USED_MEM_HIGH
}

CUDA_ARRAY_SPARSE_PROPERTIES :: struct {
    tileExtent:        struct {
        /**< Width of sparse tile in elements */
        width:  c.uint,
        /**< Height of sparse tile in elements */
        height: c.uint,
        /**< Depth of sparse tile in elements */
        depth:  c.uint,
    },
    /** First mip level at which the mip tail begins. */
    miptailFirstLevel: c.uint,
    /** Total size of the mip tail. */
    miptailSize:       c.ulonglong,
    /** Flags will either be zero or ::CU_ARRAY_SPARSE_PROPERTIES_SINGLE_MIPTAIL */
    flags:             c.uint,
    reserved:          [4]c.uint,
}

CUresourcetype :: enum(c.int) {
    ARRAY           = 0,
    MIPMAPPED_ARRAY = 1,
    LINEAR          = 2,
    PITCH2D         = 3,
}

CUDA_RESOURCE_DESC :: struct {
    /**< Resource type */
    resType: CUresourcetype,
    res:     struct #raw_union {
        array: struct {
            /**< CUDA array */
            hArray: CUarray,
        },
        mipmap: struct {
            /**< CUDA mipmapped array */
            hMipmappedArray: CUmipmappedArray,
        },
        linear: struct {
            /**< Device pointer */
            devPtr:      CUdeviceptr,
            /**< Array format */
            format:      CUarray_format,
            /**< Channels per array element */
            numChannels: c.uint,
            /**< Size in bytes */
            sizeInBytes: c.size_t,
        },
        pitch2D: struct {
            /**< Device pointer */
            devPtr:       CUdeviceptr,
            /**< Array format */
            format:       CUarray_format,
            /**< Channels per array element */
            numChannels:  c.uint,
            /**< Width of the array in elements */
            width:        c.size_t,
            /**< Height of the array in elements */
            height:       c.size_t,
            /**< Pitch between two rows in bytes */
            pitchInBytes: c.size_t,
        },
        reserved: struct {
            reserved: [32]c.int,
        },
    },
    /**< Flags (must be zero) */
    flags:   c.uint,
}

CUaddress_mode :: enum(c.int) {
    WRAP   = 0,
    CLAMP  = 1,
    MIRROR = 2,
    BORDER = 3,
}

CUfilter_mode :: enum(c.int) {
    POINT  = 0,
    LINEAR = 1,
}

CUDA_TEXTURE_DESC :: struct {
    /**< Address modes */
    addressMode:         [3]CUaddress_mode,
    /**< Filter mode */
    filterMode:          CUfilter_mode,
    /**< Flags */
    flags:               c.uint,
    /**< Maximum anisotropy ratio */
    maxAnisotropy:       c.uint,
    /**< Mipmap filter mode */
    mipmapFilterMode:    CUfilter_mode,
    /**< Mipmap level bias */
    mipmapLevelBias:     f32,
    /**< Mipmap minimum level clamp */
    minMipmapLevelClamp: f32,
    /**< Mipmap maximum level clamp */
    maxMipmapLevelClamp: f32,
    /**< Border Color */
    borderColor:         [4]f32,
    reserved:            [12]c.int,
}

CUresourceViewFormat :: enum(c.int) {
    /**< No resource view format (use underlying resource format) */
    NONE          = 0x00,
    /**< 1 channel unsigned 8-bit integers */
    UINT_1X8      = 0x01,
    /**< 2 channel unsigned 8-bit integers */
    UINT_2X8      = 0x02,
    /**< 4 channel unsigned 8-bit integers */
    UINT_4X8      = 0x03,
    /**< 1 channel signed 8-bit integers */
    SINT_1X8      = 0x04,
    /**< 2 channel signed 8-bit integers */
    SINT_2X8      = 0x05,
    /**< 4 channel signed 8-bit integers */
    SINT_4X8      = 0x06,
    /**< 1 channel unsigned 16-bit integers */
    UINT_1X16     = 0x07,
    /**< 2 channel unsigned 16-bit integers */
    UINT_2X16     = 0x08,
    /**< 4 channel unsigned 16-bit integers */
    UINT_4X16     = 0x09,
    /**< 1 channel signed 16-bit integers */
    SINT_1X16     = 0x0a,
    /**< 2 channel signed 16-bit integers */
    SINT_2X16     = 0x0b,
    /**< 4 channel signed 16-bit integers */
    SINT_4X16     = 0x0c,
    /**< 1 channel unsigned 32-bit integers */
    UINT_1X32     = 0x0d,
    /**< 2 channel unsigned 32-bit integers */
    UINT_2X32     = 0x0e,
    /**< 4 channel unsigned 32-bit integers */
    UINT_4X32     = 0x0f,
    /**< 1 channel signed 32-bit integers */
    SINT_1X32     = 0x10,
    /**< 2 channel signed 32-bit integers */
    SINT_2X32     = 0x11,
    /**< 4 channel signed 32-bit integers */
    SINT_4X32     = 0x12,
    /**< 1 channel 16-bit floating point */
    FLOAT_1X16    = 0x13,
    /**< 2 channel 16-bit floating point */
    FLOAT_2X16    = 0x14,
    /**< 4 channel 16-bit floating point */
    FLOAT_4X16    = 0x15,
    /**< 1 channel 32-bit floating point */
    FLOAT_1X32    = 0x16,
    /**< 2 channel 32-bit floating point */
    FLOAT_2X32    = 0x17,
    /**< 4 channel 32-bit floating point */
    FLOAT_4X32    = 0x18,
    /**< Block compressed 1 */
    UNSIGNED_BC1  = 0x19,
    /**< Block compressed 2 */
    UNSIGNED_BC2  = 0x1a,
    /**< Block compressed 3 */
    UNSIGNED_BC3  = 0x1b,
    /**< Block compressed 4 unsigned */
    UNSIGNED_BC4  = 0x1c,
    /**< Block compressed 4 signed */
    SIGNED_BC4    = 0x1d,
    /**< Block compressed 5 unsigned */
    UNSIGNED_BC5  = 0x1e,
    /**< Block compressed 5 signed */
    SIGNED_BC5    = 0x1f,
    /**< Block compressed 6 unsigned half-float */
    UNSIGNED_BC6H = 0x20,
    /**< Block compressed 6 signed half-float */
    SIGNED_BC6H   = 0x21,
    /**< Block compressed 7 */
    UNSIGNED_BC7  = 0x22
}

CUDA_RESOURCE_VIEW_DESC :: struct {
    /**< Resource view format */
    format:           CUresourceViewFormat,
    /**< Width of the resource view */
    width:            c.size_t,
    /**< Height of the resource view */
    height:           c.size_t,
    /**< Depth of the resource view */
    depth:            c.size_t,
    /**< First defined mipmap level */
    firstMipmapLevel: c.uint,
    /**< Last defined mipmap level */
    lastMipmapLevel:  c.uint,
    /**< First layer index */
    firstLayer:       c.uint,
    /**< Last layer index */
    lastLayer:        c.uint,
    reserved:         [16]c.uint,
}

CUdriverProcAddress_flags :: enum(c.int) {
    /**< Default search mode for driver symbols. */
    DEFAULT                   = 0,
    /**< Search for legacy versions of driver symbols. */
    LEGACY_STREAM             = 1,
    /**< Search for per-thread versions of driver symbols. */
    PER_THREAD_DEFAULT_STREAM = 2,
}

CUexecAffinityType :: enum(c.int) {
    /**< Create a context with limited SMs. */
    SM_COUNT = 0,
    MAX,
}

CUexecAffinitySmCount :: struct {
    /**< The number of SMs the context is limited to use. */
    val: c.uint,
}

CUexecAffinityParam :: struct {
    type: CUexecAffinityType,
    param: struct #raw_union {
        /** Value for ::CU_EXEC_AFFINITY_TYPE_SM_COUNT */
        smCount: CUexecAffinitySmCount,
    },
}

CUdevice_P2PAttribute :: enum(c.int) {
    /**< A relative value indicating the performance of the link between two devices */
    PERFORMANCE_RANK                     = 0x01,
    /**< P2P Access is enable */
    ACCESS_SUPPORTED                     = 0x02,
    /**< Atomic operation over the link supported */
    NATIVE_ATOMIC_SUPPORTED              = 0x03,
    /**< \deprecated use CU_DEVICE_P2P_ATTRIBUTE_CUDA_ARRAY_ACCESS_SUPPORTED instead */
    ACCESS_ACCESS_SUPPORTED              = 0x04,
    /**< Accessing CUDA arrays over the link supported */
    CUDA_ARRAY_ACCESS_SUPPORTED          = 0x04
}

CUevent_flag :: enum(c.int) {
    /**< Event uses blocking synchronization */
    BLOCKING_SYNC,
    /**< Event will not record timing data */
    DISABLE_TIMING,
    /**< Event is suitable for interprocess use. CU_EVENT_DISABLE_TIMING must be set */
    INTERPROCESS,
}
CUevent_flags :: bit_set[CUevent_flag; c.int]

CUevent_record_flags :: enum(c.int) {
    /**< Default event record flag */
    DEFAULT  = 0,
    /**< When using stream capture, create an event record node instead of the default behavior. This flag is invalid when used outside of capture. */
    EXTERNAL = 1,
}

CUevent_wait_flags :: enum(c.int) {
    /**< Default event wait flag */
    DEFAULT  = 0,
    /**< When using stream capture, create an event wait node instead of the default behavior. This flag is invalid when used outside of capture.*/
    EXTERNAL = 1,
}

CUstreamWaitValue_flags :: enum(c.int) {
    /**< Wait until (int32_t)(*addr - value) >= 0 (or int64_t for 64 bit values). Note this is a cyclic comparison which ignores wraparound. (Default behavior.) */
    GEQ   = 0,
    /**< Wait until *addr == value. */
    EQ    = 1,
    /**< Wait until (*addr & value) != 0. */
    AND   = 2,
    /**< Wait until ~(*addr | value) != 0. Support for this operation can be queried with ::cuDeviceGetAttribute() and ::CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR.*/
    NOR   = 3,
    /**< Follow the wait operation with a flush of outstanding remote writes. This means that, if a remote write operation is guaranteed to have reached the device before the wait can be satisfied, that write is guaranteed to be visible to downstream device work. The device is permitted to reorder remote writes internally. For example, this flag would be required if two remote writes arrive in a defined order, the wait is satisfied by the second write, and downstream work needs to observe the first write. Support for this operation is restricted to selected platforms and can be queried with ::CU_DEVICE_ATTRIBUTE_CAN_USE_WAIT_VALUE_FLUSH.*/
    FLUSH = 1073741824,
}

CUstreamWriteValue_flags :: enum(c.int) {
    DEFAULT           = 0,
    NO_MEMORY_BARRIER = 1,
}

CUstreamBatchMemOpType :: enum(c.int) {
    WAIT_VALUE_32       = 1,
    WRITE_VALUE_32      = 2,
    WAIT_VALUE_64       = 4,
    WRITE_VALUE_64      = 5,
    FLUSH_REMOTE_WRITES = 3,
}

CUstreamBatchMemOpParams :: struct #raw_union {
    operation: CUstreamBatchMemOpType,
    waitValue: struct {
        operation: CUstreamBatchMemOpType,
        address:   CUdeviceptr,
        using _: struct #raw_union {
            value:   u32,
            value64: u64,
        },
        flags:     c.uint,
        /**< For driver internal use. Initial value is unimportant. */
        alias:     CUdeviceptr,
    },
    writeValue: struct {
        operation: CUstreamBatchMemOpType,
        address:   CUdeviceptr,
        using _: struct #raw_union {
            value:   u32,
            value64: u64,
        },
        flags:     c.uint,
        /**< For driver internal use. Initial value is unimportant. */
        alias:     CUdeviceptr,
    },
    flushRemoteWrites: struct {
        operation: CUstreamBatchMemOpType,
        flags:     c.uint,
    },
    pad:       [6]u64,
}
/**
 * The targets for ::cuFlushGPUDirectRDMAWrites
 */
CUflushGPUDirectRDMAWritesTarget :: enum(c.int) {
    /**< Sets the target for ::cuFlushGPUDirectRDMAWrites() to the currently active CUDA device context. */
    CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TARGET_CURRENT_CTX = 0
}
/**
 * The scopes for ::cuFlushGPUDirectRDMAWrites
 */
CUflushGPUDirectRDMAWritesScope :: enum(c.int) {
    /**< Blocks until remote writes are visible to the CUDA device context owning the data. */
    CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TO_OWNER       = 100,
    /**< Blocks until remote writes are visible to all CUDA device contexts. */
    CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TO_ALL_DEVICES = 200
}
CUmemPoolPtrExportData :: struct {
    reserved: [64]c.uchar,
}

CUarraySparseSubresourceType :: enum(c.int) {
    SPARSE_LEVEL = 0,
    MIPTAIL      = 1,
}

CUmemHandleType :: enum(c.int) {
    CU_MEM_HANDLE_TYPE_GENERIC = 0,
}
CUarrayMapInfo :: struct {
    /**< Resource type */
    resourceType:     CUresourcetype,
    resource:         struct #raw_union {
        mipmap: CUmipmappedArray,
        array:  CUarray,
    },
    /**< Sparse subresource type */
    subresourceType:  CUarraySparseSubresourceType,
    subresource:      struct #raw_union {
        sparseLevel: struct {
            /**< For CUDA mipmapped arrays must a valid mipmap level. For CUDA arrays must be zero */
            level:        c.uint,
            /**< For CUDA layered arrays must be a valid layer index. Otherwise, must be zero */
            layer:        c.uint,
            /**< Starting X offset in elements */
            offsetX:      c.uint,
            /**< Starting Y offset in elements */
            offsetY:      c.uint,
            /**< Starting Z offset in elements */
            offsetZ:      c.uint,
            /**< Width in elements */
            extentWidth:  c.uint,
            /**< Height in elements */
            extentHeight: c.uint,
            /**< Depth in elements */
            extentDepth:  c.uint,
        },
        miptail: struct {
            /**< For CUDA layered arrays must be a valid layer index. Otherwise, must be zero */
            layer:  c.uint,
            /**< Offset within mip tail */
            offset: c.ulonglong,
            /**< Extent in bytes */
            size:   c.ulonglong,
        },
    },
    /**< Memory operation type */
    memOperationType: CUmemOperationType,
    /**< Memory handle type */
    memHandleType:    CUmemHandleType,
    memHandle:        struct #raw_union {
        memHandle: CUmemGenericAllocationHandle,
    },
    /**< Offset within the memory */
    offset:           c.ulonglong,
    /**< Device ordinal bit mask */
    deviceBitMask:    c.uint,
    /**< flags for future use, must be zero now. */
    flags:            c.uint,
    /**< Reserved for future use, must be zero now. */
    reserved:         [2]c.uint,
}

CUexternalMemoryHandleType :: enum c.int {
    OPAQUE_FD          = 1,
    OPAQUE_WIN32       = 2,
    OPAQUE_WIN32_KMT   = 3,
    D3D12_HEAP         = 4,
    D3D12_RESOURCE     = 5,
    D3D11_RESOURCE     = 6,
    D3D11_RESOURCE_KMT = 7,
    NVSCIBUF           = 8,
    DMABUF_FD          = 9,

}

CUDA_EXTERNAL_MEMORY_HANDLE_DESC :: struct {
    type:     CUexternalMemoryHandleType,
    handle:   struct #raw_union {
        fd:             linux.Fd,
        win32:          struct {
            handle: rawptr,
            name:   rawptr,
        },
        nvSciBufObject: rawptr,
    },
    size:     c.ulonglong,
    flags:    c.uint,
    reserved: [16]c.uint,
}

CUDA_EXTERNAL_MEMORY_BUFFER_DESC :: struct {
    offset:   c.ulonglong,
    size:     c.ulonglong,
    flags:    c.uint,
    reserved: [16]c.uint,
}

CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC :: struct {
    offset:    c.ulonglong,
    arrayDesc: CUDA_ARRAY3D_DESCRIPTOR,
    numLevels: c.uint,
    reserved:  [16]c.uint,
}

CUexternalSemaphoreHandleType :: enum(c.int) {
    OPAQUE_FD                = 1,
    OPAQUE_WIN32             = 2,
    OPAQUE_WIN32_KMT         = 3,
    D3D12_FENCE              = 4,
    D3D11_FENCE              = 5,
    NVSCISYNC                = 6,
    D3D11_KEYED_MUTEX        = 7,
    D3D11_KEYED_MUTEX_KMT    = 8,
    TIMELINE_SEMAPHORE_FD    = 9,
    TIMELINE_SEMAPHORE_WIN32 = 10,
}

CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC :: struct {
    type:     CUexternalSemaphoreHandleType,
    handle:   struct #raw_union {
        fd:           linux.Fd,
        win32:        struct {
            handle: rawptr,
            name:   rawptr,
        },
        nvSciSyncObj: rawptr,
    },
    flags:    c.uint,
    reserved: [16]c.uint,
}

when (ODIN_OS == .Windows) {
    foreign import lib "system:libcuda.lib"
} else {
    foreign import lib "system:libcuda.so"
}
count := 0

allocations : map[CUdeviceptr]runtime.Source_Code_Location

//utility function for detecting memory leaks
cuMemAlloc_v2_tracked :: proc (dptr: ^CUdeviceptr, bytesize: int, tracker:=#caller_location) -> CUresult {
    count += 1
    fmt.printfln("allocate %d", count)
    ret := cuMemAlloc_v2(dptr, bytesize)
    if ret == .SUCCESS {
        allocations[dptr^] = tracker
    }
    fmt.printfln("allocations = %v", allocations)
    return ret
}

cuMemFree_v2_tracked :: proc (dptr: CUdeviceptr) -> CUresult {
    count -= 1
    fmt.printfln("free %d", count)
    if dptr in allocations {
        delete_key(&allocations, dptr)
    }
    return cuMemFree_v2(dptr)
}

CUmemAllocationCompType :: enum c.int {
    /**< Allocating non-compressible memory */
    CU_MEM_ALLOCATION_COMP_NONE = 0x0,
    /**< Allocating  compressible memory */
    CU_MEM_ALLOCATION_COMP_GENERIC = 0x1
}

@(default_calling_convention = "stdcall" when (ODIN_OS == .Windows) else "c")
foreign lib {
    //SETUP
    cuGetErrorString  :: proc(error: CUresult, pStr: ^cstring) -> CUresult ---
    cuInit :: proc(flags: c.uint) -> CUresult ---
    cuDeviceGet :: proc(devicePtr: ^CUdevice, ordinal: i32) -> CUresult ---
    cuCtxCreate_v2 :: proc(ctx: ^CUcontext, flags: CUctx_flags, device: CUdevice) -> CUresult ---
    cuDeviceGetCount :: proc(count: ^c.int) -> CUresult ---
    cuStreamCreate :: proc(phStream: ^CUstream, flags: CUstream_flags) -> CUresult ---
    cuStreamSynchronize :: proc(hStream: CUstream) -> CUresult ---
    cuStreamDestroy :: proc(hStream: CUstream) -> CUresult ---

    //MEMORY OPERATIONS
    cuMemAlloc_v2 :: proc(dptr: ^CUdeviceptr, bytesize: int) -> CUresult ---
    cuMemAllocPitch_v2  :: proc(dptr: ^CUdeviceptr, pPitch: ^int, WidthInBytes: int, Height: int, ElementSizeBytes: c.uint) -> CUresult ---
    cuMemAllocHost_v2  :: proc(pp: ^CUdeviceptr, bytesize: int) -> CUresult ---

    cuMemFree_v2 :: proc(dptr: CUdeviceptr) -> CUresult ---

    cuMemGetInfo_v2  :: proc(free: ^int, total: ^int) -> CUresult ---
    cuMemGetAddressRange_v2  :: proc(pbase: ^CUdeviceptr, psize: ^int, dptr:CUdeviceptr) -> CUresult ---

    cuMemcpyAtoH_v2 :: proc(dstHost: ^u8, srcArray: CUarray, srcOffset: int, ByteCount: int) -> CUresult ---
    cuMemcpyDtoH_v2 :: proc(dstHost: ^u8, srcDevice: CUdeviceptr, ByteCount: int) -> CUresult ---
    cuMemcpyHtoA_v2 :: proc(dstArray: CUarray, dstOffset: int, srcHost: ^u8, ByteCount: int) -> CUresult ---
    cuMemcpyAtoA_v2 :: proc(dstArray: CUarray, dstOffset: int, srcArray: CUarray, srcOffset: int, ByteCount: int) -> CUresult ---
    cuMemcpyDtoA_v2 :: proc(dstArray: CUarray, dstOffset: int, srcDevice: CUdeviceptr, ByteCount: int) -> CUresult ---
    cuMemcpyHtoD_v2 :: proc(dstDevice: CUdeviceptr, srcHost: ^u8, ByteCount: int) -> CUresult ---
    cuMemcpyAtoD_v2 :: proc(dstDevice: CUdeviceptr, srcArray: CUarray, srcOffset: int, ByteCount: int) -> CUresult ---
    cuMemcpyDtoD_v2 :: proc(dstDevice: CUdeviceptr, srcDevice: CUdeviceptr, ByteCount: int) -> CUresult ---
    // no cuMemcpyHtoH because that's just ordinary memcpy.

    //MODULES AND KERNELS

    /** The file should be a \e cubin file as output by \b nvcc, or a \e PTX file either as output by \b nvcc or handwritten, or a \e fatbin file as output by \b nvcc from toolchain 4.0 or later.*/
    cuModuleLoad  :: proc(module: ^CUmodule, fname: cstring) -> CUresult ---
    /** Takes a pointer \p image and loads the corresponding module \p module into
        * the current context. The pointer may be obtained by mapping a \e cubin or
        * \e PTX or \e fatbin file, passing a \e cubin or \e PTX or \e fatbin file
        * as a NULL-terminated text string, or incorporating a \e cubin or \e fatbin
        * object into the executable resources and using operating system calls such
        * as Windows \c FindResource() to obtain the pointer. */
    cuModuleLoadData  :: proc(module: ^CUmodule, image: ^u8) -> CUresult ---
    cuModuleLoadDataEx  :: proc(module: ^CUmodule, image: ^u8, numOptions: c.uint, options: ^CUjit_option, optionValues: ^rawptr) -> CUresult ---
    cuModuleLoadFatBinary  :: proc(module: ^CUmodule, fatCubin: rawptr) -> CUresult ---
    cuModuleUnload  :: proc(hmod: CUmodule) -> CUresult ---
    cuModuleGetFunction  :: proc(hfunc: ^CUfunction, hmod: CUmodule, name: cstring) -> CUresult ---
    cuModuleGetTexRef  :: proc(pTexRef: ^CUtexref, hmod: CUmodule, name: cstring) -> CUresult ---
    cuModuleGetSurfRef  :: proc(pSurfRef: ^CUsurfref, hmod: CUmodule, name: cstring) -> CUresult ---

    cuModuleGetGlobal_v2  :: proc(dptr: ^CUdeviceptr, bytes: ^int, hmod: CUmodule, name: cstring) -> CUresult ---

    cuLaunchKernel  :: proc(f: CUfunction, gridDimX: c.uint, gridDimY: c.uint, gridDimZ: c.uint, blockDimX: c.uint, blockDimY: c.uint, blockDimZ: c.uint, sharedMemBytes: c.uint, hStream: CUstream, kernelParams: [^]rawptr, extra: [^]rawptr) -> CUresult ---
    cuLaunchCooperativeKernel  :: proc(f: CUfunction, gridDimX: c.uint, gridDimY: c.uint, gridDimZ: c.uint, blockDimX: c.uint, blockDimY: c.uint, blockDimZ: c.uint, sharedMemBytes: c.uint, hStream: CUstream, kernelParams: [^]rawptr) -> CUresult ---
    cuLaunchCooperativeKernelMultiDevice  :: proc(launchParamsList: ^CUDA_LAUNCH_PARAMS, numDevices: c.uint, flags: c.uint) -> CUresult ---
    cuLaunchHostFunc  :: proc(hStream: CUstream, fn: proc "c" (userData: rawptr), userData: rawptr) -> CUresult ---
    cuLaunch  :: proc(f: CUfunction) -> CUresult ---
    cuLaunchGrid  :: proc(f: CUfunction, grid_width: c.int, grid_height: c.int) -> CUresult ---
    cuLaunchGridAsync  :: proc(f: CUfunction, grid_width: c.int, grid_height: c.int, hStream: CUstream) -> CUresult ---
    cuFuncGetAttribute  :: proc(pi: ^c.int, attrib: CUfunction_attribute, hfunc: CUfunction) -> CUresult ---
    cuFuncSetAttribute  :: proc(hfunc: CUfunction, attrib: CUfunction_attribute, value: c.int) -> CUresult ---
    cuFuncSetCacheConfig  :: proc(hfunc: CUfunction, config: CUfunc_cache) -> CUresult ---
    cuFuncSetSharedMemConfig  :: proc(hfunc: CUfunction, config: CUsharedconfig) -> CUresult ---
    cuFuncGetModule  :: proc(hmod: ^CUmodule, hfunc: CUfunction) -> CUresult ---
    cuFuncSetBlockShape  :: proc(hfunc: CUfunction, x: c.int, y: c.int, z: c.int) -> CUresult ---
    cuFuncSetSharedSize  :: proc(hfunc: CUfunction, bytes: c.uint) -> CUresult ---
    cuParamSetSize  :: proc(hfunc: CUfunction, numbytes: c.uint) -> CUresult ---
    cuParamSeti  :: proc(hfunc: CUfunction, offset: c.int, value: c.uint) -> CUresult ---
    cuParamSetf  :: proc(hfunc: CUfunction, offset: c.int, value: f32) -> CUresult ---
    cuParamSetv  :: proc(hfunc: CUfunction, offset: c.int, ptr: rawptr, numbytes: c.uint) -> CUresult ---
    cuParamSetTexRef  :: proc(hfunc: CUfunction, texunit: c.int, hTexRef: CUtexref) -> CUresult ---

    cuMemcpyAsync :: proc(dst: CUdeviceptr, src: CUdeviceptr, ByteCount: c.size_t, hStream: CUstream) -> CUresult ---
    cuMemFreeAsync :: proc(dptr: CUdeviceptr, hStream: CUstream) -> CUresult ---
    cuMemAllocAsync :: proc(dptr: ^CUdeviceptr, bytesize: c.size_t, hStream: CUstream) -> CUresult ---
    cuMemcpyAtoHAsync_v2 :: proc(dstHost: ^u8, srcArray: CUarray, srcOffset: int, ByteCount: int, hStream: CUstream) -> CUresult ---
    cuMemcpyDtoHAsync_v2 :: proc(dstHost: ^u8, srcDevice: CUdeviceptr, ByteCount: int, hStream: CUstream) -> CUresult ---
    cuMemcpyHtoAAsync_v2 :: proc(dstArray: CUarray, dstOffset: int, srcHost: ^u8, ByteCount: int, hStream: CUstream) -> CUresult ---
    cuMemcpyAtoAAsync_v2 :: proc(dstArray: CUarray, dstOffset: int, srcArray: CUarray, srcOffset: int, ByteCount: int, hStream: CUstream) -> CUresult ---
    cuMemcpyDtoAAsync_v2 :: proc(dstArray: CUarray, dstOffset: int, srcDevice: CUdeviceptr, ByteCount: int, hStream: CUstream) -> CUresult ---
    cuMemcpyHtoDAsync_v2 :: proc(dstDevice: CUdeviceptr, srcHost: ^u8, ByteCount: int, hStream: CUstream) -> CUresult ---
    cuMemcpyAtoDAsync_v2 :: proc(dstDevice: CUdeviceptr, srcArray: CUarray, srcOffset: int, ByteCount: int, hStream: CUstream) -> CUresult ---
    cuMemcpyDtoDAsync_v2 :: proc(dstDevice: CUdeviceptr, srcDevice: CUdeviceptr, ByteCount: int, hStream: CUstream) -> CUresult ---
    cuMemHostGetDevicePointer_v2 :: proc(pdptr: ^CUdeviceptr, p: rawptr, flags: c.int) -> CUresult ---

    cuMemsetD8_v2 :: proc(dstDevice: CUdeviceptr, uc: u8, N: c.size_t) -> CUresult ---
    cuMemsetD16_v2 :: proc(dstDevice: CUdeviceptr, us: u16, N: c.size_t) -> CUresult ---
    cuMemsetD32_v2 :: proc(dstDevice: CUdeviceptr, ui: u32, N: c.size_t) -> CUresult ---
    cuMemsetD2D8_v2 :: proc(dstDevice: CUdeviceptr, dstPitch: c.size_t, uc: u8, Width: c.size_t, Height: c.size_t) -> CUresult ---
    cuMemsetD2D16_v2 :: proc(dstDevice: CUdeviceptr, dstPitch: c.size_t, us: u16, Width: c.size_t, Height: c.size_t) -> CUresult ---
    cuMemsetD2D32_v2 :: proc(dstDevice: CUdeviceptr, dstPitch: c.size_t, ui: u32, Width: c.size_t, Height: c.size_t) -> CUresult ---
    cuMemsetD8Async :: proc(dstDevice: CUdeviceptr, uc: u8, N: c.size_t, hStream: CUstream) -> CUresult ---
    cuMemsetD16Async :: proc(dstDevice: CUdeviceptr, us: u16, N: c.size_t, hStream: CUstream) -> CUresult ---
    cuMemsetD32Async :: proc(dstDevice: CUdeviceptr, ui: u32, N: c.size_t, hStream: CUstream) -> CUresult ---
    cuMemsetD2D8Async :: proc(dstDevice: CUdeviceptr, dstPitch: c.size_t, uc: u8, Width: c.size_t, Height: c.size_t, hStream: CUstream) -> CUresult ---
    cuMemsetD2D16Async :: proc(dstDevice: CUdeviceptr, dstPitch: c.size_t, us: u16, Width: c.size_t, Height: c.size_t, hStream: CUstream) -> CUresult ---
    cuMemsetD2D32Async :: proc(dstDevice: CUdeviceptr, dstPitch: c.size_t, ui: u32, Width: c.size_t, Height: c.size_t, hStream: CUstream) -> CUresult ---

    cuStreamBeginCapture :: proc(hStream: CUstream) -> CUresult ---
    cuStreamBeginCapture_ptsz :: proc(hStream: CUstream) -> CUresult ---
    cuStreamBeginCapture_v2 :: proc(hStream: CUstream, mode: CUstreamCaptureMode) -> CUresult ---
    cuStreamEndCapture :: proc(hStream: CUstream, phGraph: ^CUgraph) -> CUresult ---
    cuStreamIsCapturing :: proc(hStream: CUstream, captureStatus: ^CUstreamCaptureStatus) -> CUresult ---
    cuStreamGetCaptureInfo :: proc(hStream: CUstream, captureStatus_out: ^CUstreamCaptureStatus, id_out: ^u64) -> CUresult ---
    cuStreamGetCaptureInfo_v2 :: proc(hStream: CUstream, captureStatus_out: ^CUstreamCaptureStatus, id_out: ^u64, graph_out: ^CUgraph, dependencies_out:^[^]CUgraphNode, numDependencies_out:^c.size_t) -> CUresult ---
    cuStreamCopyAttributes :: proc(dstStream: CUstream, srcStream: CUstream) -> CUresult ---
    cuStreamGetAttribute :: proc(hStream: CUstream, attr: CUstreamAttrID, value: ^CUstreamAttrValue) -> CUresult ---
    cuStreamSetAttribute :: proc(hStream: CUstream, attr: CUstreamAttrID, param: ^CUstreamAttrValue) -> CUresult ---

    cuGraphCreate :: proc(phGraph: ^CUgraph, flags: c.uint) -> CUresult ---
    cuGraphAddKernelNode :: proc(phGraphNode: ^CUgraphNode, hGraph: CUgraph, dependencies: ^CUgraphNode, numDependencies: c.size_t, nodeParams: ^CUDA_KERNEL_NODE_PARAMS) -> CUresult ---
    cuGraphKernelNodeGetParams :: proc(hNode: CUgraphNode, nodeParams: ^CUDA_KERNEL_NODE_PARAMS) -> CUresult ---
    cuGraphKernelNodeSetParams :: proc(hNode: CUgraphNode, nodeParams: ^CUDA_KERNEL_NODE_PARAMS) -> CUresult ---
    cuGraphAddMemcpyNode :: proc(phGraphNode: ^CUgraphNode, hGraph: CUgraph, dependencies: ^CUgraphNode, numDependencies: c.size_t, copyParams: ^CUDA_MEMCPY3D, ctx: CUcontext) -> CUresult ---
    cuGraphMemcpyNodeGetParams :: proc(hNode: CUgraphNode, nodeParams: ^CUDA_MEMCPY3D) -> CUresult ---
    cuGraphMemcpyNodeSetParams :: proc(hNode: CUgraphNode, nodeParams: ^CUDA_MEMCPY3D) -> CUresult ---
    cuGraphAddMemsetNode :: proc(phGraphNode: ^CUgraphNode, hGraph: CUgraph, dependencies: ^CUgraphNode, numDependencies: c.size_t, memsetParams: ^CUDA_MEMSET_NODE_PARAMS, ctx: CUcontext) -> CUresult ---
    cuGraphMemsetNodeGetParams :: proc(hNode: CUgraphNode, nodeParams: ^CUDA_MEMSET_NODE_PARAMS) -> CUresult ---
    cuGraphMemsetNodeSetParams :: proc(hNode: CUgraphNode, nodeParams: ^CUDA_MEMSET_NODE_PARAMS) -> CUresult ---
    cuGraphAddHostNode :: proc(phGraphNode: ^CUgraphNode, hGraph: CUgraph, dependencies: ^CUgraphNode, numDependencies: c.size_t, nodeParams: ^CUDA_HOST_NODE_PARAMS) -> CUresult ---
    cuGraphHostNodeGetParams :: proc(hNode: CUgraphNode, nodeParams: ^CUDA_HOST_NODE_PARAMS) -> CUresult ---
    cuGraphHostNodeSetParams :: proc(hNode: CUgraphNode, nodeParams: ^CUDA_HOST_NODE_PARAMS) -> CUresult ---
    cuGraphAddChildGraphNode :: proc(phGraphNode: ^CUgraphNode, hGraph: CUgraph, dependencies: ^CUgraphNode, numDependencies: c.size_t, childGraph: CUgraph) -> CUresult ---
    cuGraphChildGraphNodeGetGraph :: proc(hNode: CUgraphNode, phGraph: ^CUgraph) -> CUresult ---
    cuGraphAddEmptyNode :: proc(phGraphNode: ^CUgraphNode, hGraph: CUgraph, dependencies: ^CUgraphNode, numDependencies: c.size_t) -> CUresult ---
    cuGraphAddEventRecordNode :: proc(phGraphNode: ^CUgraphNode, hGraph: CUgraph, dependencies: ^CUgraphNode, numDependencies: c.size_t, event: CUevent) -> CUresult ---
    cuGraphEventRecordNodeGetEvent :: proc(hNode: CUgraphNode, event_out: ^CUevent) -> CUresult ---
    cuGraphEventRecordNodeSetEvent :: proc(hNode: CUgraphNode, event: CUevent) -> CUresult ---
    cuGraphAddEventWaitNode :: proc(phGraphNode: ^CUgraphNode, hGraph: CUgraph, dependencies: ^CUgraphNode, numDependencies: c.size_t, event: CUevent) -> CUresult ---
    cuGraphEventWaitNodeGetEvent :: proc(hNode: CUgraphNode, event_out: ^CUevent) -> CUresult ---
    cuGraphEventWaitNodeSetEvent :: proc(hNode: CUgraphNode, event: CUevent) -> CUresult ---
    cuGraphAddExternalSemaphoresSignalNode :: proc(phGraphNode: ^CUgraphNode, hGraph: CUgraph, dependencies: ^CUgraphNode, numDependencies: c.size_t, nodeParams: ^CUDA_EXT_SEM_SIGNAL_NODE_PARAMS) -> CUresult ---
    cuGraphExternalSemaphoresSignalNodeGetParams :: proc(hNode: CUgraphNode, params_out: ^CUDA_EXT_SEM_SIGNAL_NODE_PARAMS) -> CUresult ---
    cuGraphExternalSemaphoresSignalNodeSetParams :: proc(hNode: CUgraphNode, nodeParams: ^CUDA_EXT_SEM_SIGNAL_NODE_PARAMS) -> CUresult ---
    cuGraphAddExternalSemaphoresWaitNode :: proc(phGraphNode: ^CUgraphNode, hGraph: CUgraph, dependencies: ^CUgraphNode, numDependencies: c.size_t, nodeParams: ^CUDA_EXT_SEM_WAIT_NODE_PARAMS) -> CUresult ---
    cuGraphExternalSemaphoresWaitNodeGetParams :: proc(hNode: CUgraphNode, params_out: ^CUDA_EXT_SEM_WAIT_NODE_PARAMS) -> CUresult ---
    cuGraphExternalSemaphoresWaitNodeSetParams :: proc(hNode: CUgraphNode, nodeParams: ^CUDA_EXT_SEM_WAIT_NODE_PARAMS) -> CUresult ---
    cuGraphAddMemAllocNode :: proc(phGraphNode: ^CUgraphNode, hGraph: CUgraph, dependencies: ^CUgraphNode, numDependencies: c.size_t, nodeParams: ^CUDA_MEM_ALLOC_NODE_PARAMS) -> CUresult ---
    cuGraphMemAllocNodeGetParams :: proc(hNode: CUgraphNode, params_out: ^CUDA_MEM_ALLOC_NODE_PARAMS) -> CUresult ---
    cuGraphAddMemFreeNode :: proc(phGraphNode: ^CUgraphNode, hGraph: CUgraph, dependencies: ^CUgraphNode, numDependencies: c.size_t, dptr: CUdeviceptr) -> CUresult ---
    cuGraphMemFreeNodeGetParams :: proc(hNode: CUgraphNode, dptr_out: ^CUdeviceptr) -> CUresult ---
    cuDeviceGraphMemTrim :: proc(device: CUdevice) -> CUresult ---
    cuDeviceGetGraphMemAttribute :: proc(device: CUdevice, attr: CUgraphMem_attribute, value: rawptr) -> CUresult ---
    cuDeviceSetGraphMemAttribute :: proc(device: CUdevice, attr: CUgraphMem_attribute, value: rawptr) -> CUresult ---
    cuGraphClone :: proc(phGraphClone: ^CUgraph, originalGraph: CUgraph) -> CUresult ---
    cuGraphNodeFindInClone :: proc(phNode: ^CUgraphNode, hOriginalNode: CUgraphNode, hClonedGraph: CUgraph) -> CUresult ---
    cuGraphNodeGetType :: proc(hNode: CUgraphNode, type: ^CUgraphNodeType) -> CUresult ---
    cuGraphGetNodes :: proc(hGraph: CUgraph, nodes: ^CUgraphNode, numNodes: ^c.size_t) -> CUresult ---
    cuGraphGetRootNodes :: proc(hGraph: CUgraph, rootNodes: ^CUgraphNode, numRootNodes: ^c.size_t) -> CUresult ---
    cuGraphGetEdges :: proc(hGraph: CUgraph, from: ^CUgraphNode, to: ^CUgraphNode, numEdges: ^c.size_t) -> CUresult ---
    cuGraphNodeGetDependencies :: proc(hNode: CUgraphNode, dependencies: ^CUgraphNode, numDependencies: ^c.size_t) -> CUresult ---
    cuGraphNodeGetDependentNodes :: proc(hNode: CUgraphNode, dependentNodes: ^CUgraphNode, numDependentNodes: ^c.size_t) -> CUresult ---
    cuGraphAddDependencies :: proc(hGraph: CUgraph, from: ^CUgraphNode, to: ^CUgraphNode, numDependencies: c.size_t) -> CUresult ---
    cuGraphRemoveDependencies :: proc(hGraph: CUgraph, from: ^CUgraphNode, to: ^CUgraphNode, numDependencies: c.size_t) -> CUresult ---
    cuGraphDestroyNode :: proc(hNode: CUgraphNode) -> CUresult ---
    cuGraphInstantiateWithFlags :: proc(phGraphExec: ^CUgraphExec, hGraph: CUgraph, flags: c.ulonglong) -> CUresult ---
    cuGraphExecKernelNodeSetParams :: proc(hGraphExec: CUgraphExec, hNode: CUgraphNode, nodeParams: ^CUDA_KERNEL_NODE_PARAMS) -> CUresult ---
    cuGraphExecMemcpyNodeSetParams :: proc(hGraphExec: CUgraphExec, hNode: CUgraphNode, copyParams: ^CUDA_MEMCPY3D, ctx: CUcontext) -> CUresult ---
    cuGraphExecMemsetNodeSetParams :: proc(hGraphExec: CUgraphExec, hNode: CUgraphNode, memsetParams: ^CUDA_MEMSET_NODE_PARAMS, ctx: CUcontext) -> CUresult ---
    cuGraphExecHostNodeSetParams :: proc(hGraphExec: CUgraphExec, hNode: CUgraphNode, nodeParams: ^CUDA_HOST_NODE_PARAMS) -> CUresult ---
    cuGraphExecChildGraphNodeSetParams :: proc(hGraphExec: CUgraphExec, hNode: CUgraphNode, childGraph: CUgraph) -> CUresult ---
    cuGraphExecEventRecordNodeSetEvent :: proc(hGraphExec: CUgraphExec, hNode: CUgraphNode, event: CUevent) -> CUresult ---
    cuGraphExecEventWaitNodeSetEvent :: proc(hGraphExec: CUgraphExec, hNode: CUgraphNode, event: CUevent) -> CUresult ---
    cuGraphExecExternalSemaphoresSignalNodeSetParams :: proc(hGraphExec: CUgraphExec, hNode: CUgraphNode, nodeParams: ^CUDA_EXT_SEM_SIGNAL_NODE_PARAMS) -> CUresult ---
    cuGraphExecExternalSemaphoresWaitNodeSetParams :: proc(hGraphExec: CUgraphExec, hNode: CUgraphNode, nodeParams: ^CUDA_EXT_SEM_WAIT_NODE_PARAMS) -> CUresult ---
    cuGraphUpload :: proc(hGraphExec: CUgraphExec, hStream: CUstream) -> CUresult ---
    cuGraphLaunch :: proc(hGraphExec: CUgraphExec, hStream: CUstream) -> CUresult ---
    cuGraphExecDestroy :: proc(hGraphExec: CUgraphExec) -> CUresult ---
    cuGraphDestroy :: proc(hGraph: CUgraph) -> CUresult ---
    cuGraphExecUpdate :: proc(hGraphExec: CUgraphExec, hGraph: CUgraph, hErrorNode_out: ^CUgraphNode, updateResult_out: ^CUgraphExecUpdateResult) -> CUresult ---
    cuGraphKernelNodeCopyAttributes :: proc(dst: CUgraphNode, src: CUgraphNode) -> CUresult ---
    cuGraphKernelNodeGetAttribute :: proc(hNode: CUgraphNode, attr: CUkernelNodeAttrID, value_out: ^CUkernelNodeAttrValue) -> CUresult ---
    cuGraphKernelNodeSetAttribute :: proc(hNode: CUgraphNode, attr: CUkernelNodeAttrID, value: ^CUkernelNodeAttrValue) -> CUresult ---
    cuGraphDebugDotPrint :: proc(hGraph: CUgraph, path: cstring, flags: c.uint) -> CUresult ---
    cuUserObjectCreate :: proc(object_out: ^CUuserObject, ptr: rawptr, destroy: proc "c" (rawptr), initialRefcount: c.uint, flags: c.uint) -> CUresult ---
    cuUserObjectRetain :: proc(object: CUuserObject, count: c.uint) -> CUresult ---
    cuUserObjectRelease :: proc(object: CUuserObject, count: c.uint) -> CUresult ---
    cuGraphRetainUserObject :: proc(graph: CUgraph, object: CUuserObject, count: c.uint, flags: c.uint) -> CUresult ---
    cuGraphReleaseUserObject :: proc(graph: CUgraph, object: CUuserObject, count: c.uint) -> CUresult ---

    cuArrayCreate_v2 :: proc(pHandle: ^CUarray, pAllocateArray: ^CUDA_ARRAY_DESCRIPTOR) -> CUresult ---
    cuArrayGetDescriptor_v2 :: proc(pArrayDescriptor: ^CUDA_ARRAY_DESCRIPTOR, hArray: CUarray) -> CUresult ---
    cuArray3DCreate_v2 :: proc(pHandle: ^CUarray, pAllocateArray: ^CUDA_ARRAY3D_DESCRIPTOR) -> CUresult ---
    cuArray3DGetDescriptor_v2 :: proc(pArrayDescriptor: ^CUDA_ARRAY3D_DESCRIPTOR, hArray: CUarray) -> CUresult ---
    cuGraphicsResourceGetMappedPointer_v2 :: proc(pDevPtr: ^CUdeviceptr, pSize: ^c.size_t, resource: CUgraphicsResource) -> CUresult ---
    cuEventDestroy_v2 :: proc(hEvent: CUevent) -> CUresult ---
    cuLinkCreate_v2 :: proc(numOptions: c.uint, options: [^]CUjit_option, optionValues: [^]rawptr, stateOut: ^CUlinkState) -> CUresult ---
    cuLinkAddData_v2 :: proc(state: CUlinkState, type: CUjitInputType, data:rawptr, size: c.size_t, name:cstring, numOptions: c.uint, options: [^]CUjit_option, optionValues: [^]rawptr) -> CUresult ---
    cuLinkAddFile :: proc(state: CUlinkState, type: CUjitInputType, path:cstring, numOptions: c.uint, options: [^]CUjit_option, optionValues: [^]rawptr) -> CUresult ---
    cuMemHostRegister_v2 :: proc(p: rawptr, bytesize: c.size_t, Flags: bit_set[CU_MEMHOSTREGISTER; c.uint]) -> CUresult ---
    cuGraphicsResourceSetMapFlags_v2 :: proc(resource: CUgraphicsResource, flags: CUgraphicsMapResourceFlags) -> CUresult ---

    cuGetErrorName                                       :: proc(error: CUresult, pStr: ^cstring) -> CUresult ---
    cuDriverGetVersion                                   :: proc(driverVersion: ^c.int) -> CUresult ---
    cuDeviceGetName                                      :: proc(name: [^]c.char, len: c.int, dev: CUdevice) -> CUresult ---
    cuDeviceGetUuid                                      :: proc(uuid: ^CUuuid, dev: CUdevice) -> CUresult ---
    cuDeviceGetUuid_v2                                   :: proc(uuid: ^CUuuid, dev: CUdevice) -> CUresult ---
    cuDeviceGetLuid                                      :: proc(luid: ^[8]c.char, deviceNodeMask: ^c.uint, dev: CUdevice) -> CUresult ---
    cuDeviceTotalMem_v2                                  :: proc(bytes: ^c.size_t, dev: CUdevice) -> CUresult ---
    cuDeviceGetTexture1DLinearMaxWidth                   :: proc(maxWidthInElements: ^c.size_t, format: CUarray_format, numChannels: c.uint, dev: CUdevice) -> CUresult ---
    cuDeviceGetAttribute                                 :: proc(pi: ^c.int, attrib: CUdevice_attribute, dev: CUdevice) -> CUresult ---
    cuDeviceGetNvSciSyncAttributes                       :: proc(nvSciSyncAttrList: rawptr, dev: CUdevice, flags: c.int) -> CUresult ---
    cuDeviceSetMemPool                                   :: proc(dev: CUdevice, pool: CUmemoryPool) -> CUresult ---
    cuDeviceGetMemPool                                   :: proc(pool: ^CUmemoryPool, dev: CUdevice) -> CUresult ---
    cuDeviceGetDefaultMemPool                            :: proc(pool_out: ^CUmemoryPool, dev: CUdevice) -> CUresult ---
    cuFlushGPUDirectRDMAWrites                           :: proc(target: CUflushGPUDirectRDMAWritesTarget, scope: CUflushGPUDirectRDMAWritesScope) -> CUresult ---
    cuDevicePrimaryCtxRetain                             :: proc(pctx: ^CUcontext, dev: CUdevice) -> CUresult ---
    cuDevicePrimaryCtxRelease_v2                         :: proc(dev: CUdevice) -> CUresult ---
    cuDevicePrimaryCtxSetFlags_v2                        :: proc(dev: CUdevice, flags: c.uint) -> CUresult ---
    cuDevicePrimaryCtxGetState                           :: proc(dev: CUdevice, flags: ^c.uint, active: ^c.int) -> CUresult ---
    cuDevicePrimaryCtxReset_v2                           :: proc(dev: CUdevice) -> CUresult ---
    cuDeviceGetExecAffinitySupport                       :: proc(pi: ^c.int, type: CUexecAffinityType, dev: CUdevice) -> CUresult ---
    cuCtxCreate_v3                                       :: proc(pctx: ^CUcontext, paramsArray: [^]CUexecAffinityParam, numParams: c.int, flags: c.uint, dev: CUdevice) -> CUresult ---
    cuCtxDestroy_v2                                      :: proc(ctx: CUcontext) -> CUresult ---
    cuCtxPushCurrent_v2                                  :: proc(ctx: CUcontext) -> CUresult ---
    cuCtxPopCurrent_v2                                   :: proc(pctx: ^CUcontext) -> CUresult ---
    cuCtxSetCurrent                                      :: proc(ctx: CUcontext) -> CUresult ---
    cuCtxGetCurrent                                      :: proc(pctx: ^CUcontext) -> CUresult ---
    cuCtxGetDevice                                       :: proc(device: ^CUdevice) -> CUresult ---
    cuCtxGetFlags                                        :: proc(flags: ^c.uint) -> CUresult ---
    cuCtxSynchronize                                     :: proc() -> CUresult ---
    cuCtxSetLimit                                        :: proc(limit: CUlimit, value: c.size_t) -> CUresult ---
    cuCtxGetLimit                                        :: proc(pvalue: ^c.size_t, limit: CUlimit) -> CUresult ---
    cuCtxGetCacheConfig                                  :: proc(pconfig: ^CUfunc_cache) -> CUresult ---
    cuCtxSetCacheConfig                                  :: proc(config: CUfunc_cache) -> CUresult ---
    cuCtxGetSharedMemConfig                              :: proc(pConfig: ^CUsharedconfig) -> CUresult ---
    cuCtxSetSharedMemConfig                              :: proc(config: CUsharedconfig) -> CUresult ---
    cuCtxGetApiVersion                                   :: proc(ctx: CUcontext, version: ^c.uint) -> CUresult ---
    cuCtxGetStreamPriorityRange                          :: proc(leastPriority: ^c.int, greatestPriority: ^c.int) -> CUresult ---
    cuCtxResetPersistingL2Cache                          :: proc() -> CUresult ---
    cuCtxGetExecAffinity                                 :: proc(pExecAffinity: ^CUexecAffinityParam, type: CUexecAffinityType) -> CUresult ---
    cuLinkAddFile_v2                                     :: proc(state: CUlinkState, type: CUjitInputType, path: cstring, numOptions: c.uint, options: [^]CUjit_option, optionValues: [^]rawptr) -> CUresult ---
    cuLinkComplete                                       :: proc(state: CUlinkState, cubinOut: ^rawptr, sizeOut: ^c.size_t) -> CUresult ---
    cuLinkDestroy                                        :: proc(state: CUlinkState) -> CUresult ---
    cuMemFreeHost                                        :: proc(p: rawptr) -> CUresult ---
    cuMemHostAlloc                                       :: proc(pp: ^rawptr, bytesize: c.size_t, Flags: c.uint) -> CUresult ---
    cuMemHostGetFlags                                    :: proc(pFlags: ^c.uint, p: rawptr) -> CUresult ---
    cuMemAllocManaged                                    :: proc(dptr: ^CUdeviceptr, bytesize: c.size_t, flags: bit_set[CUmemAttach; c.int]) -> CUresult ---
    cuDeviceGetByPCIBusId                                :: proc(dev: ^CUdevice, pciBusId: cstring) -> CUresult ---
    cuDeviceGetPCIBusId                                  :: proc(pciBusId: cstring, len: c.int, dev: CUdevice) -> CUresult ---
    cuIpcGetEventHandle                                  :: proc(pHandle: ^CUipcEventHandle, event: CUevent) -> CUresult ---
    cuIpcOpenEventHandle                                 :: proc(phEvent: ^CUevent, handle: CUipcEventHandle) -> CUresult ---
    cuIpcGetMemHandle                                    :: proc(pHandle: ^CUipcMemHandle, dptr: CUdeviceptr) -> CUresult ---
    cuIpcOpenMemHandle_v2                                :: proc(pdptr: ^CUdeviceptr, handle: CUipcMemHandle, Flags: CUipcMem_flags) -> CUresult ---
    cuIpcCloseMemHandle                                  :: proc(dptr: CUdeviceptr) -> CUresult ---
    cuMemHostUnregister                                  :: proc(p: rawptr) -> CUresult ---
    cuMemcpy                                             :: proc(dst: CUdeviceptr, src: CUdeviceptr, ByteCount: c.size_t) -> CUresult ---
    cuMemcpyPeer                                         :: proc(dstDevice: CUdeviceptr, dstContext: CUcontext, srcDevice: CUdeviceptr, srcContext: CUcontext, ByteCount: c.size_t) -> CUresult ---
    cuMemcpy2D_v2                                        :: proc(pCopy: ^CUDA_MEMCPY2D) -> CUresult ---
    cuMemcpy2DUnaligned_v2                               :: proc(pCopy: ^CUDA_MEMCPY2D) -> CUresult ---
    cuMemcpy3D_v2                                        :: proc(pCopy: ^CUDA_MEMCPY3D) -> CUresult ---
    cuMemcpy3DPeer                                       :: proc(pCopy: ^CUDA_MEMCPY3D_PEER) -> CUresult ---
    cuMemcpyPeerAsync                                    :: proc(dstDevice: CUdeviceptr, dstContext: CUcontext, srcDevice: CUdeviceptr, srcContext: CUcontext, ByteCount: c.size_t, hStream: CUstream) -> CUresult ---
    cuMemcpy2DAsync_v2                                   :: proc(pCopy: ^CUDA_MEMCPY2D, hStream: CUstream) -> CUresult ---
    cuMemcpy3DAsync_v2                                   :: proc(pCopy: ^CUDA_MEMCPY3D, hStream: CUstream) -> CUresult ---
    cuMemcpy3DPeerAsync                                  :: proc(pCopy: ^CUDA_MEMCPY3D_PEER, hStream: CUstream) -> CUresult ---
    cuArrayGetSparseProperties                           :: proc(sparseProperties: ^CUDA_ARRAY_SPARSE_PROPERTIES, array: CUarray) -> CUresult ---
    cuMipmappedArrayGetSparseProperties                  :: proc(sparseProperties: ^CUDA_ARRAY_SPARSE_PROPERTIES, mipmap: CUmipmappedArray) -> CUresult ---
    cuArrayGetPlane                                      :: proc(pPlaneArray: ^CUarray, hArray: CUarray, planeIdx: c.uint) -> CUresult ---
    cuArrayDestroy                                       :: proc(hArray: CUarray) -> CUresult ---
    cuMipmappedArrayCreate                               :: proc(pHandle: ^CUmipmappedArray, pMipmappedArrayDesc: ^CUDA_ARRAY3D_DESCRIPTOR, numMipmapLevels: c.uint) -> CUresult ---
    cuMipmappedArrayGetLevel                             :: proc(pLevelArray: ^CUarray, hMipmappedArray: CUmipmappedArray, level: c.uint) -> CUresult ---
    cuMipmappedArrayDestroy                              :: proc(hMipmappedArray: CUmipmappedArray) -> CUresult ---
    cuMemAddressReserve                                  :: proc(ptr: ^CUdeviceptr, size: c.size_t, alignment: c.size_t, addr: CUdeviceptr, flags: c.ulonglong) -> CUresult ---
    cuMemAddressFree                                     :: proc(ptr: CUdeviceptr, size: c.size_t) -> CUresult ---
    cuMemCreate                                          :: proc(handle: ^CUmemGenericAllocationHandle, size: c.size_t, prop: ^CUmemAllocationProp, flags: c.ulonglong) -> CUresult ---
    cuMemRelease                                         :: proc(handle: CUmemGenericAllocationHandle) -> CUresult ---
    cuMemMap                                             :: proc(ptr: CUdeviceptr, size: c.size_t, offset: c.size_t, handle: CUmemGenericAllocationHandle, flags: c.ulonglong) -> CUresult ---
    cuMemMapArrayAsync                                   :: proc(mapInfoList: ^CUarrayMapInfo, count: c.uint, hStream: CUstream) -> CUresult ---
    cuMemUnmap                                           :: proc(ptr: CUdeviceptr, size: c.size_t) -> CUresult ---
    cuMemSetAccess                                       :: proc(ptr: CUdeviceptr, size: c.size_t, desc: ^CUmemAccessDesc, count: c.size_t) -> CUresult ---
    cuMemGetAccess                                       :: proc(flags: ^c.ulonglong, location: ^CUmemLocation, ptr: CUdeviceptr) -> CUresult ---
    cuMemExportToShareableHandle                         :: proc(shareableHandle: rawptr, handle: CUmemGenericAllocationHandle, handleType: CUmemAllocationHandleType, flags: c.ulonglong) -> CUresult ---
    cuMemImportFromShareableHandle                       :: proc(handle: ^CUmemGenericAllocationHandle, osHandle: rawptr, shHandleType: CUmemAllocationHandleType) -> CUresult ---
    cuMemGetAllocationGranularity                        :: proc(granularity: ^c.size_t, prop: ^CUmemAllocationProp, option: CUmemAllocationGranularity_flags) -> CUresult ---
    cuMemGetAllocationPropertiesFromHandle               :: proc(prop: ^CUmemAllocationProp, handle: CUmemGenericAllocationHandle) -> CUresult ---
    cuMemRetainAllocationHandle                          :: proc(handle: ^CUmemGenericAllocationHandle, addr: rawptr) -> CUresult ---
    cuMemPoolTrimTo                                      :: proc(pool: CUmemoryPool, minBytesToKeep: c.size_t) -> CUresult ---
    cuMemPoolSetAttribute                                :: proc(pool: CUmemoryPool, attr: CUmemPool_attribute, value: rawptr) -> CUresult ---
    cuMemPoolGetAttribute                                :: proc(pool: CUmemoryPool, attr: CUmemPool_attribute, value: rawptr) -> CUresult ---
    cuMemPoolSetAccess                                   :: proc(pool: CUmemoryPool, _map: ^CUmemAccessDesc, count: c.size_t) -> CUresult ---
    cuMemPoolGetAccess                                   :: proc(flags: ^CUmemAccess_flags, memPool: CUmemoryPool, location: ^CUmemLocation) -> CUresult ---
    cuMemPoolCreate                                      :: proc(pool: ^CUmemoryPool, poolProps: ^CUmemPoolProps) -> CUresult ---
    cuMemPoolDestroy                                     :: proc(pool: CUmemoryPool) -> CUresult ---
    cuMemAllocFromPoolAsync                              :: proc(dptr: ^CUdeviceptr, bytesize: c.size_t, pool: CUmemoryPool, hStream: CUstream) -> CUresult ---
    cuMemPoolExportToShareableHandle                     :: proc(handle_out: rawptr, pool: CUmemoryPool, handleType: CUmemAllocationHandleType, flags: c.ulonglong) -> CUresult ---
    cuMemPoolImportFromShareableHandle                   :: proc(pool_out: ^CUmemoryPool, handle: rawptr, handleType: CUmemAllocationHandleType, flags: c.ulonglong) -> CUresult ---
    cuMemPoolExportPointer                               :: proc(shareData_out: ^CUmemPoolPtrExportData, ptr: CUdeviceptr) -> CUresult ---
    cuMemPoolImportPointer                               :: proc(ptr_out: ^CUdeviceptr, pool: CUmemoryPool, shareData: ^CUmemPoolPtrExportData) -> CUresult ---
    cuPointerGetAttribute                                :: proc(data: rawptr, attribute: CUpointer_attribute, ptr: CUdeviceptr) -> CUresult ---
    cuMemPrefetchAsync                                   :: proc(devPtr: CUdeviceptr, count: c.size_t, dstDevice: CUdevice, hStream: CUstream) -> CUresult ---
    cuMemAdvise                                          :: proc(devPtr: CUdeviceptr, count: c.size_t, advice: CUmem_advise, device: CUdevice) -> CUresult ---
    cuMemRangeGetAttribute                               :: proc(data: rawptr, dataSize: c.size_t, attribute: CUmem_range_attribute, devPtr: CUdeviceptr, count: c.size_t) -> CUresult ---
    cuMemRangeGetAttributes                              :: proc(data: ^rawptr, dataSizes: ^c.size_t, attributes: ^CUmem_range_attribute, numAttributes: c.size_t, devPtr: CUdeviceptr, count: c.size_t) -> CUresult ---
    cuPointerSetAttribute                                :: proc(value: rawptr, attribute: CUpointer_attribute, ptr: CUdeviceptr) -> CUresult ---
    cuPointerGetAttributes                               :: proc(numAttributes: c.uint, attributes: ^CUpointer_attribute, data: ^rawptr, ptr: CUdeviceptr) -> CUresult ---
    cuStreamCreateWithPriority                           :: proc(phStream: ^CUstream, flags: c.uint, priority: c.int) -> CUresult ---
    cuStreamGetPriority                                  :: proc(hStream: CUstream, priority: ^c.int) -> CUresult ---
    cuStreamGetFlags                                     :: proc(hStream: CUstream, flags: ^c.uint) -> CUresult ---
    cuStreamGetCtx                                       :: proc(hStream: CUstream, pctx: ^CUcontext) -> CUresult ---
    cuStreamWaitEvent                                    :: proc(hStream: CUstream, hEvent: CUevent, Flags: c.uint) -> CUresult ---
    cuStreamAddCallback                                  :: proc(hStream: CUstream, callback: proc "c" (CUstream, CUresult, rawptr), userData: rawptr, flags: c.uint) -> CUresult ---
    cuThreadExchangeStreamCaptureMode                    :: proc(mode: ^CUstreamCaptureMode) -> CUresult ---
    cuStreamUpdateCaptureDependencies                    :: proc(hStream: CUstream, dependencies: ^CUgraphNode, numDependencies: c.size_t, flags: c.uint) -> CUresult ---
    cuStreamAttachMemAsync                               :: proc(hStream: CUstream, dptr: CUdeviceptr, length: c.size_t, flags: c.uint) -> CUresult ---
    cuStreamQuery                                        :: proc(hStream: CUstream) -> CUresult ---
    cuStreamDestroy_v2                                   :: proc(hStream: CUstream) -> CUresult ---
    cuEventCreate                                        :: proc(phEvent: ^CUevent, Flags: CUevent_flags) -> CUresult ---
    cuEventRecord                                        :: proc(hEvent: CUevent, hStream: CUstream) -> CUresult ---
    cuEventRecordWithFlags                               :: proc(hEvent: CUevent, hStream: CUstream, flags: CUevent_record_flags) -> CUresult ---
    cuEventQuery                                         :: proc(hEvent: CUevent) -> CUresult ---
    cuEventSynchronize                                   :: proc(hEvent: CUevent) -> CUresult ---
    cuEventElapsedTime                                   :: proc(pMilliseconds: ^f32, hStart: CUevent, hEnd: CUevent) -> CUresult ---
    cuImportExternalMemory                               :: proc(extMem_out: ^CUexternalMemory, memHandleDesc: ^CUDA_EXTERNAL_MEMORY_HANDLE_DESC) -> CUresult ---
    cuExternalMemoryGetMappedBuffer                      :: proc(devPtr: ^CUdeviceptr, extMem: CUexternalMemory, bufferDesc: ^CUDA_EXTERNAL_MEMORY_BUFFER_DESC) -> CUresult ---
    cuExternalMemoryGetMappedMipmappedArray              :: proc(mipmap: ^CUmipmappedArray, extMem: CUexternalMemory, mipmapDesc: ^CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC) -> CUresult ---
    cuDestroyExternalMemory                              :: proc(extMem: CUexternalMemory) -> CUresult ---
    cuImportExternalSemaphore                            :: proc(extSem_out: ^CUexternalSemaphore, semHandleDesc: ^CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC) -> CUresult ---
    cuSignalExternalSemaphoresAsync                      :: proc(extSemArray: [^]CUexternalSemaphore, paramsArray: [^]CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS, numExtSems: c.uint, stream: CUstream) -> CUresult ---
    cuWaitExternalSemaphoresAsync                        :: proc(extSemArray: [^]CUexternalSemaphore, paramsArray: [^]CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS, numExtSems: c.uint, stream: CUstream) -> CUresult ---
    cuDestroyExternalSemaphore                           :: proc(extSem: CUexternalSemaphore) -> CUresult ---
    cuStreamWaitValue32                                  :: proc(stream: CUstream, addr: CUdeviceptr, value: u32, flags: c.uint) -> CUresult ---
    cuStreamWaitValue64                                  :: proc(stream: CUstream, addr: CUdeviceptr, value: u64, flags: c.uint) -> CUresult ---
    cuStreamWriteValue32                                 :: proc(stream: CUstream, addr: CUdeviceptr, value: u32, flags: c.uint) -> CUresult ---
    cuStreamWriteValue64                                 :: proc(stream: CUstream, addr: CUdeviceptr, value: u64, flags: c.uint) -> CUresult ---
    cuStreamBatchMemOp                                   :: proc(stream: CUstream, count: c.uint, paramArray: [^]CUstreamBatchMemOpParams, flags: c.uint) -> CUresult ---
    cuGraphInstantiate_v2                                :: proc(phGraphExec: ^CUgraphExec, hGraph: CUgraph, phErrorNode: ^CUgraphNode, logBuffer: cstring, bufferSize: c.size_t) -> CUresult ---
    cuOccupancyMaxActiveBlocksPerMultiprocessor          :: proc(numBlocks: ^c.int, func: CUfunction, blockSize: c.int, dynamicSMemSize: c.size_t) -> CUresult ---
    cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags :: proc(numBlocks: ^c.int, func: CUfunction, blockSize: c.int, dynamicSMemSize: c.size_t, flags: c.uint) -> CUresult ---
    cuOccupancyMaxPotentialBlockSize                     :: proc(minGridSize: ^c.int, blockSize: ^c.int, func: CUfunction, blockSizeToDynamicSMemSize: proc "c" (c.int) -> c.size_t, dynamicSMemSize: c.size_t, blockSizeLimit: c.int) -> CUresult ---
    cuOccupancyMaxPotentialBlockSizeWithFlags            :: proc(minGridSize: ^c.int, blockSize: ^c.int, func: CUfunction, blockSizeToDynamicSMemSize: proc "c" (c.int) -> c.size_t, dynamicSMemSize: c.size_t, blockSizeLimit: c.int, flags: c.uint) -> CUresult ---
    cuOccupancyAvailableDynamicSMemPerBlock              :: proc(dynamicSmemSize: ^c.size_t, func: CUfunction, numBlocks: c.int, blockSize: c.int) -> CUresult ---
    cuTexObjectCreate                                    :: proc(pTexObject: ^CUtexObject, pResDesc: ^CUDA_RESOURCE_DESC, pTexDesc: ^CUDA_TEXTURE_DESC, pResViewDesc: ^CUDA_RESOURCE_VIEW_DESC) -> CUresult ---
    cuTexObjectDestroy                                   :: proc(texObject: CUtexObject) -> CUresult ---
    cuTexObjectGetResourceDesc                           :: proc(pResDesc: ^CUDA_RESOURCE_DESC, texObject: CUtexObject) -> CUresult ---
    cuTexObjectGetTextureDesc                            :: proc(pTexDesc: ^CUDA_TEXTURE_DESC, texObject: CUtexObject) -> CUresult ---
    cuTexObjectGetResourceViewDesc                       :: proc(pResViewDesc: ^CUDA_RESOURCE_VIEW_DESC, texObject: CUtexObject) -> CUresult ---
    cuSurfObjectCreate                                   :: proc(pSurfObject: ^CUsurfObject, pResDesc: ^CUDA_RESOURCE_DESC) -> CUresult ---
    cuSurfObjectDestroy                                  :: proc(surfObject: CUsurfObject) -> CUresult ---
    cuSurfObjectGetResourceDesc                          :: proc(pResDesc: ^CUDA_RESOURCE_DESC, surfObject: CUsurfObject) -> CUresult ---
    cuDeviceCanAccessPeer                                :: proc(canAccessPeer: ^c.int, dev: CUdevice, peerDev: CUdevice) -> CUresult ---
    cuCtxEnablePeerAccess                                :: proc(peerContext: CUcontext, Flags: c.uint) -> CUresult ---
    cuCtxDisablePeerAccess                               :: proc(peerContext: CUcontext) -> CUresult ---
    cuDeviceGetP2PAttribute                              :: proc(value: ^c.int, attrib: CUdevice_P2PAttribute, srcDevice: CUdevice, dstDevice: CUdevice) -> CUresult ---
    cuGraphicsUnregisterResource                         :: proc(resource: CUgraphicsResource) -> CUresult ---
    cuGraphicsSubResourceGetMappedArray                  :: proc(pArray: ^CUarray, resource: CUgraphicsResource, arrayIndex: c.uint, mipLevel: c.uint) -> CUresult ---
    cuGraphicsResourceGetMappedMipmappedArray            :: proc(pMipmappedArray: ^CUmipmappedArray, resource: CUgraphicsResource) -> CUresult ---
    cuGraphicsMapResources                               :: proc(count: c.uint, resources: [^]CUgraphicsResource, hStream: CUstream) -> CUresult ---
    cuGraphicsUnmapResources                             :: proc(count: c.uint, resources: [^]CUgraphicsResource, hStream: CUstream) -> CUresult ---
    cuGetProcAddress                                     :: proc(symbol: cstring, pfn: ^rawptr, cudaVersion: c.int, flags: u64) -> CUresult ---
    cuGetExportTable                                     :: proc(ppExportTable: ^rawptr, pExportTableId: ^CUuuid) -> CUresult ---
}

EGLImageKHR :: rawptr
EGLStreamKHR :: rawptr
CUeglStreamConnection :: distinct rawptr
CUeglResourceLocationFlags :: enum c.int {
    /**< Resource location sysmem */
    CU_EGL_RESOURCE_LOCATION_SYSMEM   = 0x00,
    /**< Resource location vidmem */
    CU_EGL_RESOURCE_LOCATION_VIDMEM   = 0x01
}
MAX_PLANES :: 3
CUeglFrameType :: enum c.int {
    /**< Frame type CUDA array */
    CU_EGL_FRAME_TYPE_ARRAY = 0,
    /**< Frame type pointer */
    CU_EGL_FRAME_TYPE_PITCH = 1,
}

CUeglColorFormat :: enum c.int {
    /**< Y, U, V in three surfaces, each in a separate surface, U/V width = 1/2 Y width, U/V height = 1/2 Y height. */
    CU_EGL_COLOR_FORMAT_YUV420_PLANAR              = 0x00,
    /**< Y, UV in two surfaces (UV as one surface) with VU byte ordering, width, height ratio same as YUV420Planar. */
    CU_EGL_COLOR_FORMAT_YUV420_SEMIPLANAR          = 0x01,
    /**< Y, U, V  each in a separate  surface, U/V width = 1/2 Y width, U/V height = Y height. */
    CU_EGL_COLOR_FORMAT_YUV422_PLANAR              = 0x02,
    /**< Y, UV in two surfaces with VU byte ordering, width, height ratio same as YUV422Planar. */
    CU_EGL_COLOR_FORMAT_YUV422_SEMIPLANAR          = 0x03,
    /**< R/G/B three channels in one surface with BGR byte ordering. Only pitch linear format supported. */
    CU_EGL_COLOR_FORMAT_RGB                        = 0x04,
    /**< R/G/B three channels in one surface with RGB byte ordering. Only pitch linear format supported. */
    CU_EGL_COLOR_FORMAT_BGR                        = 0x05,
    /**< R/G/B/A four channels in one surface with BGRA byte ordering. */
    CU_EGL_COLOR_FORMAT_ARGB                       = 0x06,
    /**< R/G/B/A four channels in one surface with ABGR byte ordering. */
    CU_EGL_COLOR_FORMAT_RGBA                       = 0x07,
    /**< single luminance channel in one surface. */
    CU_EGL_COLOR_FORMAT_L                          = 0x08,
    /**< single color channel in one surface. */
    CU_EGL_COLOR_FORMAT_R                          = 0x09,
    /**< Y, U, V in three surfaces, each in a separate surface, U/V width = Y width, U/V height = Y height. */
    CU_EGL_COLOR_FORMAT_YUV444_PLANAR              = 0x0A,
    /**< Y, UV in two surfaces (UV as one surface) with VU byte ordering, width, height ratio same as YUV444Planar. */
    CU_EGL_COLOR_FORMAT_YUV444_SEMIPLANAR          = 0x0B,
    /**< Y, U, V in one surface, interleaved as UYVY in one channel. */
    CU_EGL_COLOR_FORMAT_YUYV_422                   = 0x0C,
    /**< Y, U, V in one surface, interleaved as YUYV in one channel. */
    CU_EGL_COLOR_FORMAT_UYVY_422                   = 0x0D,
    /**< R/G/B/A four channels in one surface with RGBA byte ordering. */
    CU_EGL_COLOR_FORMAT_ABGR                       = 0x0E,
    /**< R/G/B/A four channels in one surface with ARGB byte ordering. */
    CU_EGL_COLOR_FORMAT_BGRA                       = 0x0F,
    /**< Alpha color format - one channel in one surface. */
    CU_EGL_COLOR_FORMAT_A                          = 0x10,
    /**< R/G color format - two channels in one surface with GR byte ordering */
    CU_EGL_COLOR_FORMAT_RG                         = 0x11,
    /**< Y, U, V, A four channels in one surface, interleaved as VUYA. */
    CU_EGL_COLOR_FORMAT_AYUV                       = 0x12,
    /**< Y, VU in two surfaces (VU as one surface) with UV byte ordering, U/V width = Y width, U/V height = Y height. */
    CU_EGL_COLOR_FORMAT_YVU444_SEMIPLANAR          = 0x13,
    /**< Y, VU in two surfaces (VU as one surface) with UV byte ordering, U/V width = 1/2 Y width, U/V height = Y height. */
    CU_EGL_COLOR_FORMAT_YVU422_SEMIPLANAR          = 0x14,
    /**< Y, VU in two surfaces (VU as one surface) with UV byte ordering, U/V width = 1/2 Y width, U/V height = 1/2 Y height. */
    CU_EGL_COLOR_FORMAT_YVU420_SEMIPLANAR          = 0x15,
    /**< Y10, V10U10 in two surfaces (VU as one surface) with UV byte ordering, U/V width = Y width, U/V height = Y height. */
    CU_EGL_COLOR_FORMAT_Y10V10U10_444_SEMIPLANAR   = 0x16,
    /**< Y10, V10U10 in two surfaces (VU as one surface) with UV byte ordering, U/V width = 1/2 Y width, U/V height = 1/2 Y height. */
    CU_EGL_COLOR_FORMAT_Y10V10U10_420_SEMIPLANAR   = 0x17,
    /**< Y12, V12U12 in two surfaces (VU as one surface) with UV byte ordering, U/V width = Y width, U/V height = Y height. */
    CU_EGL_COLOR_FORMAT_Y12V12U12_444_SEMIPLANAR   = 0x18,
    /**< Y12, V12U12 in two surfaces (VU as one surface) with UV byte ordering, U/V width = 1/2 Y width, U/V height = 1/2 Y height. */
    CU_EGL_COLOR_FORMAT_Y12V12U12_420_SEMIPLANAR   = 0x19,
    /**< Extended Range Y, U, V in one surface, interleaved as YVYU in one channel. */
    CU_EGL_COLOR_FORMAT_VYUY_ER                    = 0x1A,
    /**< Extended Range Y, U, V in one surface, interleaved as YUYV in one channel. */
    CU_EGL_COLOR_FORMAT_UYVY_ER                    = 0x1B,
    /**< Extended Range Y, U, V in one surface, interleaved as UYVY in one channel. */
    CU_EGL_COLOR_FORMAT_YUYV_ER                    = 0x1C,
    /**< Extended Range Y, U, V in one surface, interleaved as VYUY in one channel. */
    CU_EGL_COLOR_FORMAT_YVYU_ER                    = 0x1D,
    /**< Extended Range Y, U, V three channels in one surface, interleaved as VUY. Only pitch linear format supported. */
    CU_EGL_COLOR_FORMAT_YUV_ER                     = 0x1E,
    /**< Extended Range Y, U, V, A four channels in one surface, interleaved as AVUY. */
    CU_EGL_COLOR_FORMAT_YUVA_ER                    = 0x1F,
    /**< Extended Range Y, U, V, A four channels in one surface, interleaved as VUYA. */
    CU_EGL_COLOR_FORMAT_AYUV_ER                    = 0x20,
    /**< Extended Range Y, U, V in three surfaces, U/V width = Y width, U/V height = Y height. */
    CU_EGL_COLOR_FORMAT_YUV444_PLANAR_ER           = 0x21,
    /**< Extended Range Y, U, V in three surfaces, U/V width = 1/2 Y width, U/V height = Y height. */
    CU_EGL_COLOR_FORMAT_YUV422_PLANAR_ER           = 0x22,
    /**< Extended Range Y, U, V in three surfaces, U/V width = 1/2 Y width, U/V height = 1/2 Y height. */
    CU_EGL_COLOR_FORMAT_YUV420_PLANAR_ER           = 0x23,
    /**< Extended Range Y, UV in two surfaces (UV as one surface) with VU byte ordering, U/V width = Y width, U/V height = Y height. */
    CU_EGL_COLOR_FORMAT_YUV444_SEMIPLANAR_ER       = 0x24,
    /**< Extended Range Y, UV in two surfaces (UV as one surface) with VU byte ordering, U/V width = 1/2 Y width, U/V height = Y height. */
    CU_EGL_COLOR_FORMAT_YUV422_SEMIPLANAR_ER       = 0x25,
    /**< Extended Range Y, UV in two surfaces (UV as one surface) with VU byte ordering, U/V width = 1/2 Y width, U/V height = 1/2 Y height. */
    CU_EGL_COLOR_FORMAT_YUV420_SEMIPLANAR_ER       = 0x26,
    /**< Extended Range Y, V, U in three surfaces, U/V width = Y width, U/V height = Y height. */
    CU_EGL_COLOR_FORMAT_YVU444_PLANAR_ER           = 0x27,
    /**< Extended Range Y, V, U in three surfaces, U/V width = 1/2 Y width, U/V height = Y height. */
    CU_EGL_COLOR_FORMAT_YVU422_PLANAR_ER           = 0x28,
    /**< Extended Range Y, V, U in three surfaces, U/V width = 1/2 Y width, U/V height = 1/2 Y height. */
    CU_EGL_COLOR_FORMAT_YVU420_PLANAR_ER           = 0x29,
    /**< Extended Range Y, VU in two surfaces (VU as one surface) with UV byte ordering, U/V width = Y width, U/V height = Y height. */
    CU_EGL_COLOR_FORMAT_YVU444_SEMIPLANAR_ER       = 0x2A,
    /**< Extended Range Y, VU in two surfaces (VU as one surface) with UV byte ordering, U/V width = 1/2 Y width, U/V height = Y height. */
    CU_EGL_COLOR_FORMAT_YVU422_SEMIPLANAR_ER       = 0x2B,
    /**< Extended Range Y, VU in two surfaces (VU as one surface) with UV byte ordering, U/V width = 1/2 Y width, U/V height = 1/2 Y height. */
    CU_EGL_COLOR_FORMAT_YVU420_SEMIPLANAR_ER       = 0x2C,
    /**< Bayer format - one channel in one surface with interleaved RGGB ordering. */
    CU_EGL_COLOR_FORMAT_BAYER_RGGB                 = 0x2D,
    /**< Bayer format - one channel in one surface with interleaved BGGR ordering. */
    CU_EGL_COLOR_FORMAT_BAYER_BGGR                 = 0x2E,
    /**< Bayer format - one channel in one surface with interleaved GRBG ordering. */
    CU_EGL_COLOR_FORMAT_BAYER_GRBG                 = 0x2F,
    /**< Bayer format - one channel in one surface with interleaved GBRG ordering. */
    CU_EGL_COLOR_FORMAT_BAYER_GBRG                 = 0x30,
    /**< Bayer10 format - one channel in one surface with interleaved RGGB ordering. Out of 16 bits, 10 bits used 6 bits No-op. */
    CU_EGL_COLOR_FORMAT_BAYER10_RGGB               = 0x31,
    /**< Bayer10 format - one channel in one surface with interleaved BGGR ordering. Out of 16 bits, 10 bits used 6 bits No-op. */
    CU_EGL_COLOR_FORMAT_BAYER10_BGGR               = 0x32,
    /**< Bayer10 format - one channel in one surface with interleaved GRBG ordering. Out of 16 bits, 10 bits used 6 bits No-op. */
    CU_EGL_COLOR_FORMAT_BAYER10_GRBG               = 0x33,
    /**< Bayer10 format - one channel in one surface with interleaved GBRG ordering. Out of 16 bits, 10 bits used 6 bits No-op. */
    CU_EGL_COLOR_FORMAT_BAYER10_GBRG               = 0x34,
    /**< Bayer12 format - one channel in one surface with interleaved RGGB ordering. Out of 16 bits, 12 bits used 4 bits No-op. */
    CU_EGL_COLOR_FORMAT_BAYER12_RGGB               = 0x35,
    /**< Bayer12 format - one channel in one surface with interleaved BGGR ordering. Out of 16 bits, 12 bits used 4 bits No-op. */
    CU_EGL_COLOR_FORMAT_BAYER12_BGGR               = 0x36,
    /**< Bayer12 format - one channel in one surface with interleaved GRBG ordering. Out of 16 bits, 12 bits used 4 bits No-op. */
    CU_EGL_COLOR_FORMAT_BAYER12_GRBG               = 0x37,
    /**< Bayer12 format - one channel in one surface with interleaved GBRG ordering. Out of 16 bits, 12 bits used 4 bits No-op. */
    CU_EGL_COLOR_FORMAT_BAYER12_GBRG               = 0x38,
    /**< Bayer14 format - one channel in one surface with interleaved RGGB ordering. Out of 16 bits, 14 bits used 2 bits No-op. */
    CU_EGL_COLOR_FORMAT_BAYER14_RGGB               = 0x39,
    /**< Bayer14 format - one channel in one surface with interleaved BGGR ordering. Out of 16 bits, 14 bits used 2 bits No-op. */
    CU_EGL_COLOR_FORMAT_BAYER14_BGGR               = 0x3A,
    /**< Bayer14 format - one channel in one surface with interleaved GRBG ordering. Out of 16 bits, 14 bits used 2 bits No-op. */
    CU_EGL_COLOR_FORMAT_BAYER14_GRBG               = 0x3B,
    /**< Bayer14 format - one channel in one surface with interleaved GBRG ordering. Out of 16 bits, 14 bits used 2 bits No-op. */
    CU_EGL_COLOR_FORMAT_BAYER14_GBRG               = 0x3C,
    /**< Bayer20 format - one channel in one surface with interleaved RGGB ordering. Out of 32 bits, 20 bits used 12 bits No-op. */
    CU_EGL_COLOR_FORMAT_BAYER20_RGGB               = 0x3D,
    /**< Bayer20 format - one channel in one surface with interleaved BGGR ordering. Out of 32 bits, 20 bits used 12 bits No-op. */
    CU_EGL_COLOR_FORMAT_BAYER20_BGGR               = 0x3E,
    /**< Bayer20 format - one channel in one surface with interleaved GRBG ordering. Out of 32 bits, 20 bits used 12 bits No-op. */
    CU_EGL_COLOR_FORMAT_BAYER20_GRBG               = 0x3F,
    /**< Bayer20 format - one channel in one surface with interleaved GBRG ordering. Out of 32 bits, 20 bits used 12 bits No-op. */
    CU_EGL_COLOR_FORMAT_BAYER20_GBRG               = 0x40,
    /**< Y, V, U in three surfaces, each in a separate surface, U/V width = Y width, U/V height = Y height. */
    CU_EGL_COLOR_FORMAT_YVU444_PLANAR              = 0x41,
    /**< Y, V, U in three surfaces, each in a separate surface, U/V width = 1/2 Y width, U/V height = Y height. */
    CU_EGL_COLOR_FORMAT_YVU422_PLANAR              = 0x42,
    /**< Y, V, U in three surfaces, each in a separate surface, U/V width = 1/2 Y width, U/V height = 1/2 Y height. */
    CU_EGL_COLOR_FORMAT_YVU420_PLANAR              = 0x43,
    /**< Nvidia proprietary Bayer ISP format - one channel in one surface with interleaved RGGB ordering and mapped to opaque integer datatype. */
    CU_EGL_COLOR_FORMAT_BAYER_ISP_RGGB             = 0x44,
    /**< Nvidia proprietary Bayer ISP format - one channel in one surface with interleaved BGGR ordering and mapped to opaque integer datatype. */
    CU_EGL_COLOR_FORMAT_BAYER_ISP_BGGR             = 0x45,
    /**< Nvidia proprietary Bayer ISP format - one channel in one surface with interleaved GRBG ordering and mapped to opaque integer datatype. */
    CU_EGL_COLOR_FORMAT_BAYER_ISP_GRBG             = 0x46,
    /**< Nvidia proprietary Bayer ISP format - one channel in one surface with interleaved GBRG ordering and mapped to opaque integer datatype. */
    CU_EGL_COLOR_FORMAT_BAYER_ISP_GBRG             = 0x47,
    /**< Bayer format - one channel in one surface with interleaved BCCR ordering. */
    CU_EGL_COLOR_FORMAT_BAYER_BCCR                 = 0x48,
    /**< Bayer format - one channel in one surface with interleaved RCCB ordering. */
    CU_EGL_COLOR_FORMAT_BAYER_RCCB                 = 0x49,
    /**< Bayer format - one channel in one surface with interleaved CRBC ordering. */
    CU_EGL_COLOR_FORMAT_BAYER_CRBC                 = 0x4A,
    /**< Bayer format - one channel in one surface with interleaved CBRC ordering. */
    CU_EGL_COLOR_FORMAT_BAYER_CBRC                 = 0x4B,
    /**< Bayer10 format - one channel in one surface with interleaved CCCC ordering. Out of 16 bits, 10 bits used 6 bits No-op. */
    CU_EGL_COLOR_FORMAT_BAYER10_CCCC               = 0x4C,
    /**< Bayer12 format - one channel in one surface with interleaved BCCR ordering. Out of 16 bits, 12 bits used 4 bits No-op. */
    CU_EGL_COLOR_FORMAT_BAYER12_BCCR               = 0x4D,
    /**< Bayer12 format - one channel in one surface with interleaved RCCB ordering. Out of 16 bits, 12 bits used 4 bits No-op. */
    CU_EGL_COLOR_FORMAT_BAYER12_RCCB               = 0x4E,
    /**< Bayer12 format - one channel in one surface with interleaved CRBC ordering. Out of 16 bits, 12 bits used 4 bits No-op. */
    CU_EGL_COLOR_FORMAT_BAYER12_CRBC               = 0x4F,
    /**< Bayer12 format - one channel in one surface with interleaved CBRC ordering. Out of 16 bits, 12 bits used 4 bits No-op. */
    CU_EGL_COLOR_FORMAT_BAYER12_CBRC               = 0x50,
    /**< Bayer12 format - one channel in one surface with interleaved CCCC ordering. Out of 16 bits, 12 bits used 4 bits No-op. */
    CU_EGL_COLOR_FORMAT_BAYER12_CCCC               = 0x51,
    /**< Color format for single Y plane. */
    CU_EGL_COLOR_FORMAT_Y                          = 0x52,
    /**< Y, UV in two surfaces (UV as one surface) U/V width = 1/2 Y width, U/V height = 1/2 Y height. */
    CU_EGL_COLOR_FORMAT_YUV420_SEMIPLANAR_2020     = 0x53,
    /**< Y, VU in two surfaces (VU as one surface) U/V width = 1/2 Y width, U/V height = 1/2 Y height. */
    CU_EGL_COLOR_FORMAT_YVU420_SEMIPLANAR_2020     = 0x54,
    /**< Y, U, V  each in a separate  surface, U/V width = 1/2 Y width, U/V height= 1/2 Y height. */             
    CU_EGL_COLOR_FORMAT_YUV420_PLANAR_2020         = 0x55,
    /**< Y, V, U each in a separate surface, U/V width = 1/2 Y width, U/V height = 1/2 Y height. */
    CU_EGL_COLOR_FORMAT_YVU420_PLANAR_2020         = 0x56,
    /**< Y, UV in two surfaces (UV as one surface) U/V width = 1/2 Y width, U/V height = 1/2 Y height. */
    CU_EGL_COLOR_FORMAT_YUV420_SEMIPLANAR_709      = 0x57,
    /**< Y, VU in two surfaces (VU as one surface) U/V width = 1/2 Y width, U/V height = 1/2 Y height. */
    CU_EGL_COLOR_FORMAT_YVU420_SEMIPLANAR_709      = 0x58,
    /**< Y, U, V  each in a separate  surface, U/V width = 1/2 Y width, U/V height = 1/2 Y height. */
    CU_EGL_COLOR_FORMAT_YUV420_PLANAR_709          = 0x59,
    /**< Y, V, U each in a separate surface, U/V width = 1/2 Y width, U/V height = 1/2 Y height. */
    CU_EGL_COLOR_FORMAT_YVU420_PLANAR_709          = 0x5A,
    /**< Y10, V10U10 in two surfaces (VU as one surface), U/V width = 1/2 Y width, U/V height = 1/2 Y height. */
    CU_EGL_COLOR_FORMAT_Y10V10U10_420_SEMIPLANAR_709  = 0x5B,
    /**< Y10, V10U10 in two surfaces (VU as one surface), U/V width = 1/2 Y width, U/V height = 1/2 Y height. */
    CU_EGL_COLOR_FORMAT_Y10V10U10_420_SEMIPLANAR_2020 = 0x5C,
    /**< Y10, V10U10 in two surfaces(VU as one surface) U/V width = 1/2 Y width, U/V height  = Y height. */
    CU_EGL_COLOR_FORMAT_Y10V10U10_422_SEMIPLANAR_2020 = 0x5D,
    /**< Y10, V10U10 in two surfaces(VU as one surface) U/V width = 1/2 Y width, U/V height  = Y height. */
    CU_EGL_COLOR_FORMAT_Y10V10U10_422_SEMIPLANAR      = 0x5E,
    /**< Y10, V10U10 in two surfaces(VU as one surface) U/V width = 1/2 Y width, U/V height  = Y height. */
    CU_EGL_COLOR_FORMAT_Y10V10U10_422_SEMIPLANAR_709  = 0x5F,
    /**< Extended Range Color format for single Y plane. */
    CU_EGL_COLOR_FORMAT_Y_ER                          = 0x60,
    /**< Extended Range Color format for single Y plane. */
    CU_EGL_COLOR_FORMAT_Y_709_ER                      = 0x61,
    /**< Extended Range Color format for single Y10 plane. */
    CU_EGL_COLOR_FORMAT_Y10_ER                        = 0x62,
    /**< Extended Range Color format for single Y10 plane. */
    CU_EGL_COLOR_FORMAT_Y10_709_ER                    = 0x63,
    /**< Extended Range Color format for single Y12 plane. */
    CU_EGL_COLOR_FORMAT_Y12_ER                        = 0x64,
    /**< Extended Range Color format for single Y12 plane. */
    CU_EGL_COLOR_FORMAT_Y12_709_ER                    = 0x65,
    /**< Y, U, V, A four channels in one surface, interleaved as AVUY. */
    CU_EGL_COLOR_FORMAT_YUVA                          = 0x66,
    /**< Y, U, V three channels in one surface, interleaved as VUY. Only pitch linear format supported. */
    CU_EGL_COLOR_FORMAT_YUV                           = 0x67,
    /**< Y, U, V in one surface, interleaved as YVYU in one channel. */
    CU_EGL_COLOR_FORMAT_YVYU                          = 0x68,
    /**< Y, U, V in one surface, interleaved as VYUY in one channel. */
    CU_EGL_COLOR_FORMAT_VYUY                          = 0x69,
    /**< Extended Range Y10, V10U10 in two surfaces(VU as one surface) U/V width = 1/2 Y width, U/V height = 1/2 Y height. */
    CU_EGL_COLOR_FORMAT_Y10V10U10_420_SEMIPLANAR_ER     = 0x6A,
    /**< Extended Range Y10, V10U10 in two surfaces(VU as one surface) U/V width = 1/2 Y width, U/V height = 1/2 Y height. */
    CU_EGL_COLOR_FORMAT_Y10V10U10_420_SEMIPLANAR_709_ER = 0x6B,
    /**< Extended Range Y10, V10U10 in two surfaces (VU as one surface) U/V width = Y width, U/V height = Y height. */ 
    CU_EGL_COLOR_FORMAT_Y10V10U10_444_SEMIPLANAR_ER     = 0x6C,
    /**< Extended Range Y10, V10U10 in two surfaces (VU as one surface)  U/V width = Y width, U/V height = Y height. */
    CU_EGL_COLOR_FORMAT_Y10V10U10_444_SEMIPLANAR_709_ER = 0x6D,
    /**< Extended Range Y12, V12U12 in two surfaces (VU as one surface) U/V width = 1/2 Y width, U/V height = 1/2 Y height. */ 
    CU_EGL_COLOR_FORMAT_Y12V12U12_420_SEMIPLANAR_ER     = 0x6E,
    /**< Extended Range Y12, V12U12 in two surfaces (VU as one surface) U/V width = 1/2 Y width, U/V height = 1/2 Y height. */
    CU_EGL_COLOR_FORMAT_Y12V12U12_420_SEMIPLANAR_709_ER = 0x6F,
    /**< Extended Range Y12, V12U12 in two surfaces (VU as one surface) U/V width = Y width, U/V height = Y height. */ 
    CU_EGL_COLOR_FORMAT_Y12V12U12_444_SEMIPLANAR_ER     = 0x70,
    /**< Extended Range Y12, V12U12 in two surfaces (VU as one surface) U/V width = Y width, U/V height = Y height. */
    CU_EGL_COLOR_FORMAT_Y12V12U12_444_SEMIPLANAR_709_ER = 0x71,
    /**< Y, U, V in one surface, interleaved as UYVY in one channel. */
    CU_EGL_COLOR_FORMAT_UYVY_709                        = 0x72,
    /**< Extended Range Y, U, V in one surface, interleaved as UYVY in one channel. */
    CU_EGL_COLOR_FORMAT_UYVY_709_ER                     = 0x73,
    /**< Y, U, V in one surface, interleaved as UYVY in one channel. */
    CU_EGL_COLOR_FORMAT_UYVY_2020                       = 0x74,
    CU_EGL_COLOR_FORMAT_MAX
}

CUeglFrame :: struct {
    frame: struct #raw_union {
        /**< Array of CUarray corresponding to each plane*/
        pArray: [MAX_PLANES]CUarray,
        /**< Array of Pointers corresponding to each plane*/
        pPitch: [MAX_PLANES]rawptr,
    },
    /**< Width of first plane */
    width: c.uint,
    /**< Height of first plane */
    height: c.uint,
    /**< Depth of first plane */
    depth: c.uint,
    /**< Pitch of first plane */
    pitch: c.uint,
    /**< Number of planes */
    planeCount: c.uint,
    /**< Number of channels for the plane */
    numChannels: c.uint,
    /**< Array or Pitch */
    frameType: CUeglFrameType,
    /**< CUDA EGL Color Format*/
    eglColorFormat: CUeglColorFormat,
    /**< CUDA Array Format*/
    cuFormat: CUarray_format,
}

CUgraphicsRegisterFlags :: enum c.uint { 
    READ_ONLY,
    WRITE_DISCARD,
    SURFACE_LDST,
    TEXTURE_GATHER,
}

//Not part of cuda.h
@(default_calling_convention = "stdcall" when (ODIN_OS == .Windows) else "c")
foreign lib {
    cuGraphicsEGLRegisterImage                           :: proc(pCudaResource: ^CUgraphicsResource, image: EGLImageKHR, flags:CUgraphicsMapResourceFlags) -> CUresult ---                                        
    cuEGLStreamConsumerConnect                           :: proc(conn: ^CUeglStreamConnection, stream: EGLStreamKHR) -> CUresult ---
    cuEGLStreamConsumerConnectWithFlags                  :: proc(conn: ^CUeglStreamConnection, stream: EGLStreamKHR, flags: CUeglResourceLocationFlags) -> CUresult ---
    cuEGLStreamConsumerDisconnect                        :: proc(conn: ^CUeglStreamConnection) -> CUresult ---
    cuEGLStreamConsumerAcquireFrame                      :: proc(conn: ^CUeglStreamConnection, pCudaResource: ^CUgraphicsResource, pStream: ^CUstream, timeout_micros: c.uint) -> CUresult ---
    cuEGLStreamConsumerReleaseFrame                      :: proc(conn: ^CUeglStreamConnection, pCudaResource: CUgraphicsResource, pStream: ^CUstream) -> CUresult ---
    
    cuEGLStreamProducerConnect                           :: proc(conn: ^CUeglStreamConnection, stream: EGLStreamKHR, width, height: i32) -> CUresult ---
    cuEGLStreamProducerDisconnect                        :: proc(conn: ^CUeglStreamConnection) -> CUresult ---
    cuEGLStreamProducerPresentFrame                      :: proc(conn: ^CUeglStreamConnection, eglframe: CUeglFrame, pStream: ^CUstream) -> CUresult ---
    cuEGLStreamProducerReturnFrame                       :: proc(conn: ^CUeglStreamConnection, eglframe: ^CUeglFrame, pStream: ^CUstream) -> CUresult ---
    cuGraphicsResourceGetMappedEglFrame                  :: proc(eglframe: ^CUeglFrame, resource: CUgraphicsResource, index: c.uint, mipLevel: c.uint) -> CUresult ---

    cuGraphicsGLRegisterBuffer                           :: proc(pCudaResource: ^CUgraphicsResource, buffer: u32, flags: bit_set[CUgraphicsRegisterFlags; c.uint]) -> CUresult ---
    cuGraphicsGLRegisterImage                            :: proc(pCudaResource: ^CUgraphicsResource, image: u32, target_GLenum: i32, flags: bit_set[CUgraphicsRegisterFlags; c.uint]) -> CUresult ---
}

@(require_results)
/** NON-CUDA util function. Copies the given values into GPU memory, and returns a list of deviceptrs pointing to them.
 * Cuda-Freeing the first returned pointer will free them all. The slice itself must be freed separately.
 * The returned pointers are guaranteed to be a multiple of **align** bytes apart, though the alignment of the first returned pointer is up to cuda.
 */
util_copy_from_device :: proc(ptr: CUdeviceptr, $T:typeid) -> ^T {
    res := new(T)
    cuMemcpyDtoH_v2(auto_cast res, ptr, size_of(T))
    return res
}

@(require_results)
/** NON-CUDA util function. Copies the given values into GPU memory, and returns a list of deviceptrs pointing to them.
 * Cuda-Freeing the first returned pointer will free them all. The slice itself must be freed separately.
 * The returned pointers are guaranteed to be a multiple of **align** bytes apart, though the alignment of the first returned pointer is up to cuda.
 */
util_copy_to_device :: proc(align:int, data: ..any) ->  ([]CUdeviceptr, CUresult) {
    full_size := 0

    pointers := make([]CUdeviceptr, len(data))
    for var in data {
        full_size += reflect.size_of_typeid(var.id)
        full_size = runtime.align_forward(full_size, align)
    }
    ptr: CUdeviceptr
    res := cuMemAlloc_v2(&ptr, full_size)
    if res != .SUCCESS {
        return pointers, res
    }
    full_size = 0

    for var, ind in data {
        pointers[ind] = ptr + CUdeviceptr(full_size)
        data_size := reflect.size_of_typeid(var.id)
        res = cuMemcpyHtoD_v2(pointers[ind] , auto_cast var.data, data_size)
        full_size += data_size
        full_size = runtime.align_forward(full_size, align)
        if res != .SUCCESS {
            return pointers, res
        }
    }
    return pointers, res
}

enable_formatting :: proc "odin" () {
    fmt.register_user_formatter(type_info_of(CUdeviceptr).id, proc(info: ^fmt.Info, arg: any, format: rune) -> bool {
        switch v in arg {
            case CUdeviceptr:
                switch format {
                    case 'x': fallthrough
                    case 'p':
                        io.write_string(info.writer, fmt.tprintf("&device[%x]", uint(v)))
                        return true
                    case 'X': fallthrough
                    case 'P': fallthrough
                    case 'V': fallthrough
                    case 'v':
                        io.write_string(info.writer, fmt.tprintf("&device[%X]", uint(v)))
                        return true
                    case 'd': fallthrough
                    case 'D':
                        io.write_string(info.writer, fmt.tprintf("&device[%d]", uint(v)))
                        return true
                }
        }
        return false
    })
}

make_extra :: proc(extras: ^[6]rawptr, args: ^$T) {
    //unfortunately we can't just get the size_of(T) because that would be too easy.
    //cuda refuses to launch the kernel if the size of the buffer isn't exactly the size of the USED area in the buffer
    //so padding must be removed from the end.
    structInfo := reflect.type_info_base(type_info_of(T)).variant.(reflect.Type_Info_Struct)
    last := structInfo.field_count - 1
    offset := structInfo.offsets[last]
    last_size := uintptr((structInfo.types[last]).size)
    extras^ = [6]rawptr{
        CU_LAUNCH_PARAM_BUFFER_POINTER, args,
        CU_LAUNCH_PARAM_BUFFER_SIZE, &extras[5],
        CU_LAUNCH_PARAM_END, rawptr(offset + last_size)
    }
}


cu_assert :: proc(x: $T, exp:=#caller_expression, loc:=#caller_location) -> T {
    assert(x == CUresult.SUCCESS, fmt.tprintfln("%v @ %s", x, exp), loc)
    return x
}
