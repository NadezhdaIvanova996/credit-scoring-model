from onnxruntime.quantization import quantize_dynamic, QuantType
import os

# Пути к моделям
original_onnx = "models/model.onnx"
quantized_onnx = "models/model_quant.onnx"

# Делаем динамическую квантизацию (на 8-bit)
quantize_dynamic(original_onnx, quantized_onnx, weight_type=QuantType.QUInt8)

# Выводим размеры для отчёта
original_size = os.path.getsize(original_onnx) / (1024 * 1024)  # MB
quant_size = os.path.getsize(quantized_onnx) / (1024 * 1024)     # MB

print(f"Quantization завершена! Квантизованная модель сохранена: {quantized_onnx}")
print(f"Размер оригинальной ONNX: {original_size:.3f} MB")
print(f"Размер квантизованной модели: {quant_size:.3f} MB")
print(f"Уменьшение размера: {original_size / quant_size:.1f}x")