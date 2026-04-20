/*
 * onnx_reader.h — ONNX float32 tensor reader
 */
#ifndef BONFYRE_ONNX_READER_H
#define BONFYRE_ONNX_READER_H

#include <stdint.h>
#include <stddef.h>

#define ONNX_NAME_MAX 256

typedef struct {
    char    name[ONNX_NAME_MAX];
    float  *data;        /* float32 row-major, caller frees */
    size_t  n_elements;
} OnnxTensor;

/*
 * onnx_read — parse all float32 initializer tensors from an ONNX model.
 * Returns malloc'd array of OnnxTensor; *n_out is length.
 * On error returns NULL.  Caller frees with onnx_tensors_free().
 */
OnnxTensor *onnx_read(const char *path, size_t *n_out);

void onnx_tensors_free(OnnxTensor *tensors, size_t n);

#endif /* BONFYRE_ONNX_READER_H */
