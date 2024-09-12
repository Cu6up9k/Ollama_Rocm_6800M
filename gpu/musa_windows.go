package gpu

import (
	"errors"
	"log/slog"
)

// Gather GPU information from the mtgpu driver if any supported GPUs are detected
func MUSAGetGPUInfo() []MusaGPUInfo {
	slog.Warn("Unsupported platform")
	return []MusaGPUInfo{}
}

func (gpus MusaGPUInfoList) RefreshFreeMemory() error {
	return errors.New("unsupported platform")
}
