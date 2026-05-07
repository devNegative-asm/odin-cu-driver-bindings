package nvrtc
import "core:c"
import "core:fmt"

NvrtcResult :: enum (i32) {
    SUCCESS,
    ERROR_OUT_OF_MEMORY,
    ERROR_PROGRAM_CREATION_FAILURE,
    ERROR_INVALID_INPUT,
    ERROR_INVALID_PROGRAM,
    ERROR_INVALID_OPTION,
    ERROR_COMPILATION,
    ERROR_BUILTIN_OPERATION_FAILURE,
    ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION,
    ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION,
    ERROR_NAME_EXPRESSION_NOT_VALID,
    ERROR_INTERNAL_ERROR,
}
NvrtcProgram :: distinct rawptr

foreign import lib "system:libnvrtc.so"
@(default_calling_convention = "stdcall" when (ODIN_OS == .Windows) else "c")
foreign lib {
    nvrtcVersion :: proc(major, minor: ^c.int) -> NvrtcResult ---
    /**
     * \ingroup query
     * \brief   nvrtcGetNumSupportedArchs sets the output parameter \p numArchs
     *          with the number of architectures supported by NVRTC. This can
     *          then be used to pass an array to ::nvrtcGetSupportedArchs to
     *          get the supported architectures.
     *
     * \param   [out] numArchs number of supported architectures.
     * \return
     *   - \link #NvrtcResult NVRTC_SUCCESS \endlink
     *   - \link #NvrtcResult NVRTC_ERROR_INVALID_INPUT \endlink
     *
     * see    ::nvrtcGetSupportedArchs
     */
    nvrtcGetNumSupportedArchs :: proc(numArchs: ^c.int) -> NvrtcResult ---
    /**
     * \ingroup query
     * \brief   nvrtcGetSupportedArchs populates the array passed via the output parameter
     *          \p supportedArchs with the architectures supported by NVRTC. The array is
     *          sorted in the ascending order. The size of the array to be passed can be
     *          determined using ::nvrtcGetNumSupportedArchs.
     *
     * \param   [out] supportedArchs sorted array of supported architectures.
     * \return
     *   - \link #NvrtcResult NVRTC_SUCCESS \endlink
     *   - \link #NvrtcResult NVRTC_ERROR_INVALID_INPUT \endlink
     *
     * see    ::nvrtcGetNumSupportedArchs
     */
    nvrtcGetSupportedArchs :: proc(supportedArchs: [^]c.int) -> NvrtcResult ---
    /**
     * \ingroup compilation
     * \brief   nvrtcCreateProgram creates an instance of NvrtcProgram with the
     *          given input parameters, and sets the output parameter \p prog with
     *          it.
     *
     * \param   [out] prog         CUDA Runtime Compilation program.
     * \param   [in]  src          CUDA program source.
     * \param   [in]  name         CUDA program name.\n
     *                             \p name can be \c NULL; \c "default_program" is
     *                             used when \p name is \c NULL or "".
     * \param   [in]  numHeaders   Number of headers used.\n
     *                             \p numHeaders must be greater than or equal to 0.
     * \param   [in]  headers      Sources of the headers.\n
     *                             \p headers can be \c NULL when \p numHeaders is
     *                             0.
     * \param   [in]  includeNames Name of each header by which they can be
     *                             included in the CUDA program source.\n
     *                             \p includeNames can be \c NULL when \p numHeaders
     *                             is 0.
     * \return
     *   - \link #NvrtcResult NVRTC_SUCCESS \endlink
     *   - \link #NvrtcResult NVRTC_ERROR_OUT_OF_MEMORY \endlink
     *   - \link #NvrtcResult NVRTC_ERROR_PROGRAM_CREATION_FAILURE \endlink
     *   - \link #NvrtcResult NVRTC_ERROR_INVALID_INPUT \endlink
     *   - \link #NvrtcResult NVRTC_ERROR_INVALID_PROGRAM \endlink
     *
     * \see     ::nvrtcDestroyProgram
     */
    nvrtcCreateProgram :: proc(prog: ^NvrtcProgram, src: cstring, name: cstring, numHeaders: c.int, headers_source:[^]cstring, includeNames:[^]cstring) -> NvrtcResult ---

    /**
     * \ingroup compilation
     * \brief   nvrtcDestroyProgram destroys the given program.
     *
     * \param    [in] prog CUDA Runtime Compilation program.
     * \return
     *   - \link #NvrtcResult NVRTC_SUCCESS \endlink
     *   - \link #NvrtcResult NVRTC_ERROR_INVALID_PROGRAM \endlink
     *
     * \see     ::nvrtcCreateProgram
     */
    nvrtcDestroyProgram :: proc(prog: ^NvrtcProgram) -> NvrtcResult ---
    /**
    * \ingroup compilation
    * \brief   nvrtcCompileProgram compiles the given program.
    *
    * \param   [in] prog       CUDA Runtime Compilation program.
    * \param   [in] numOptions Number of compiler options passed.
    * \param   [in] options    Compiler options in the form of C string array.\n
    *                          \p options can be \c NULL when \p numOptions is 0.
    *
    * \return
    *   - \link #NvrtcResult NVRTC_SUCCESS \endlink
    *   - \link #NvrtcResult NVRTC_ERROR_OUT_OF_MEMORY \endlink
    *   - \link #NvrtcResult NVRTC_ERROR_INVALID_INPUT \endlink
    *   - \link #NvrtcResult NVRTC_ERROR_INVALID_PROGRAM \endlink
    *   - \link #NvrtcResult NVRTC_ERROR_INVALID_OPTION \endlink
    *   - \link #NvrtcResult NVRTC_ERROR_COMPILATION \endlink
    *   - \link #NvrtcResult NVRTC_ERROR_BUILTIN_OPERATION_FAILURE \endlink
    *
    * It supports compile options listed in \ref options.
    */
    nvrtcCompileProgram :: proc(prog: NvrtcProgram, numOptions: c.int, options: [^]cstring) -> NvrtcResult ---

    /**
     * \ingroup compilation
     * \brief   nvrtcGetPTXSize sets \p ptxSizeRet with the size of the PTX
     *          generated by the previous compilation of \p prog (including the
     *          trailing \c NULL).
     *
     * \param   [in]  prog       CUDA Runtime Compilation program.
     * \param   [out] ptxSizeRet Size of the generated PTX (including the trailing
     *                           \c NULL).
     * \return
     *   - \link #NvrtcResult NVRTC_SUCCESS \endlink
     *   - \link #NvrtcResult NVRTC_ERROR_INVALID_INPUT \endlink
     *   - \link #NvrtcResult NVRTC_ERROR_INVALID_PROGRAM \endlink
     *
     * \see     ::nvrtcGetPTX
     */
    nvrtcGetPTXSize :: proc(prog: NvrtcProgram, ptxSizeRet: ^c.size_t) -> NvrtcResult ---


    /**
     * \ingroup compilation
     * \brief   nvrtcGetPTX stores the PTX generated by the previous compilation
     *          of \p prog in the memory pointed by \p ptx.
     *
     * \param   [in]  prog CUDA Runtime Compilation program.
     * \param   [out] ptx  Compiled result.
     * \return
     *   - \link #NvrtcResult NVRTC_SUCCESS \endlink
     *   - \link #NvrtcResult NVRTC_ERROR_INVALID_INPUT \endlink
     *   - \link #NvrtcResult NVRTC_ERROR_INVALID_PROGRAM \endlink
     *
     * \see     ::nvrtcGetPTXSize
     */
    nvrtcGetPTX :: proc(prog: NvrtcProgram, ptx:[^]u8) -> NvrtcResult ---


    /**
     * \ingroup compilation
     * \brief   nvrtcGetCUBINSize sets \p cubinSizeRet with the size of the cubin
     *          generated by the previous compilation of \p prog. The value of
     *          cubinSizeRet is set to 0 if the value specified to \c -arch is a
     *          virtual architecture instead of an actual architecture.
     *
     * \param   [in]  prog       CUDA Runtime Compilation program.
     * \param   [out] cubinSizeRet Size of the generated cubin.
     * \return
     *   - \link #NvrtcResult NVRTC_SUCCESS \endlink
     *   - \link #NvrtcResult NVRTC_ERROR_INVALID_INPUT \endlink
     *   - \link #NvrtcResult NVRTC_ERROR_INVALID_PROGRAM \endlink
     *
     * \see     ::nvrtcGetCUBIN
     */
    nvrtcGetCUBINSize :: proc(prog: NvrtcProgram, cubinSizeRet: ^c.size_t) -> NvrtcResult ---


    /**
     * \ingroup compilation
     * \brief   nvrtcGetCUBIN stores the cubin generated by the previous compilation
     *          of \p prog in the memory pointed by \p cubin. No cubin is available
     *          if the value specified to \c -arch is a virtual architecture instead
     *          of an actual architecture.
     *
     * \param   [in]  prog CUDA Runtime Compilation program.
     * \param   [out] cubin  Compiled and assembled result.
     * \return
     *   - \link #NvrtcResult NVRTC_SUCCESS \endlink
     *   - \link #NvrtcResult NVRTC_ERROR_INVALID_INPUT \endlink
     *   - \link #NvrtcResult NVRTC_ERROR_INVALID_PROGRAM \endlink
     *
     * \see     ::nvrtcGetCUBINSize
     */
    nvrtcGetCUBIN :: proc(prog: NvrtcProgram, cubin:[^]u8) -> NvrtcResult ---

    /**
     * \ingroup compilation
     * \brief   nvrtcGetNVVMSize sets \p nvvmSizeRet with the size of the NVVM
     *          generated by the previous compilation of \p prog. The value of
     *          nvvmSizeRet is set to 0 if the program was not compiled with
     *          \c -dlto.
     *
     * \param   [in]  prog       CUDA Runtime Compilation program.
     * \param   [out] nvvmSizeRet Size of the generated NVVM.
     * \return
     *   - \link #NvrtcResult NVRTC_SUCCESS \endlink
     *   - \link #NvrtcResult NVRTC_ERROR_INVALID_INPUT \endlink
     *   - \link #NvrtcResult NVRTC_ERROR_INVALID_PROGRAM \endlink
     *
     * \see     ::nvrtcGetNVVM
     */
    nvrtcGetNVVMSize :: proc(prog: NvrtcProgram, nvvmSizeRet:^c.size_t) -> NvrtcResult ---


    /**
     * \ingroup compilation
     * \brief   nvrtcGetNVVM stores the NVVM generated by the previous compilation
     *          of \p prog in the memory pointed by \p nvvm.
     *          The program must have been compiled with -dlto,
     *          otherwise will return an error.
     *
     * \param   [in]  prog CUDA Runtime Compilation program.
     * \param   [out] nvvm Compiled result.
     * \return
     *   - \link #NvrtcResult NVRTC_SUCCESS \endlink
     *   - \link #NvrtcResult NVRTC_ERROR_INVALID_INPUT \endlink
     *   - \link #NvrtcResult NVRTC_ERROR_INVALID_PROGRAM \endlink
     *
     * \see     ::nvrtcGetNVVMSize
     */
    nvrtcGetNVVM :: proc(prog: NvrtcProgram, nvvm:[^]u8) -> NvrtcResult ---

    /**
     * \ingroup compilation
     * \brief   nvrtcGetProgramLogSize sets \p logSizeRet with the size of the
     *          log generated by the previous compilation of \p prog (including the
     *          trailing \c NULL).
     *
     * Note that compilation log may be generated with warnings and informative
     * messages, even when the compilation of \p prog succeeds.
     *
     * \param   [in]  prog       CUDA Runtime Compilation program.
     * \param   [out] logSizeRet Size of the compilation log
     *                           (including the trailing \c NULL).
     * \return
     *   - \link #NvrtcResult NVRTC_SUCCESS \endlink
     *   - \link #NvrtcResult NVRTC_ERROR_INVALID_INPUT \endlink
     *   - \link #NvrtcResult NVRTC_ERROR_INVALID_PROGRAM \endlink
     *
     * \see     ::nvrtcGetProgramLog
     */
    nvrtcGetProgramLogSize :: proc(prog: NvrtcProgram, logSizeRet: ^c.size_t) -> NvrtcResult ---


    /**
     * \ingroup compilation
     * \brief   nvrtcGetProgramLog stores the log generated by the previous
     *          compilation of \p prog in the memory pointed by \p log.
     *
     * \param   [in]  prog CUDA Runtime Compilation program.
     * \param   [out] log  Compilation log.
     * \return
     *   - \link #NvrtcResult NVRTC_SUCCESS \endlink
     *   - \link #NvrtcResult NVRTC_ERROR_INVALID_INPUT \endlink
     *   - \link #NvrtcResult NVRTC_ERROR_INVALID_PROGRAM \endlink
     *
     * \see     ::nvrtcGetProgramLogSize
     */
    nvrtcGetProgramLog :: proc(prog: NvrtcProgram, log:[^]u8) -> NvrtcResult ---


    /**
     * \ingroup compilation
     * \brief   nvrtcAddNameExpression notes the given name expression
     *          denoting the address of a __global__ function
     *          or __device__/__constant__ variable.
     *
     * The identical name expression string must be provided on a subsequent
     * call to nvrtcGetLoweredName to extract the lowered name.
     * \param   [in]  prog CUDA Runtime Compilation program.
     * \param   [in] name_expression constant expression denoting the address of
     *               a __global__ function or __device__/__constant__ variable.
     * \return
     *   - \link #NvrtcResult NVRTC_SUCCESS \endlink
     *   - \link #NvrtcResult NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION \endlink
     *
     * \see     ::nvrtcGetLoweredName
     */
    nvrtcAddNameExpression :: proc(prog: NvrtcProgram, name_expression: cstring) -> NvrtcResult ---

    /**
     * \ingroup compilation
     * \brief   nvrtcGetLoweredName extracts the lowered (mangled) name
     *          for a __global__ function or __device__/__constant__ variable,
     *          and updates *lowered_name to point to it. The memory containing
     *          the name is released when the NVRTC program is destroyed by
     *          nvrtcDestroyProgram.
     *          The identical name expression must have been previously
     *          provided to nvrtcAddNameExpression.
     *
     * \param   [in]  prog CUDA Runtime Compilation program.
     * \param   [in] name_expression constant expression denoting the address of
     *               a __global__ function or __device__/__constant__ variable.
     * \param   [out] lowered_name initialized by the function to point to a
     *               C string containing the lowered (mangled)
     *               name corresponding to the provided name expression.
     * \return
     *   - \link #NvrtcResult NVRTC_SUCCESS \endlink
     *   - \link #NvrtcResult NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION \endlink
     *   - \link #NvrtcResult NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID \endlink
     *
     * \see     ::nvrtcAddNameExpression
     */
    nvrtcGetLoweredName :: proc(prog: NvrtcProgram, name_expression: cstring, lowered_name: ^cstring) -> NvrtcResult ---
}


nvrtc_assert :: proc(x: NvrtcResult, exp:=#caller_expression, loc:=#caller_location) -> NvrtcResult {
    assert(x == .SUCCESS, fmt.tprintfln("%v @ %s", x, exp), loc)
    return x
}