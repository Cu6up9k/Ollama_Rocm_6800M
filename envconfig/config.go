package envconfig

import (
	"fmt"
	"log/slog"
	"math"
	"net"
	"net/url"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"time"
)

// Host returns the scheme and host. Host can be configured via the OLLAMA_HOST environment variable.
// Default is scheme "http" and host "127.0.0.1:11434"
func Host() *url.URL {
	defaultPort := "11434"

	s := strings.TrimSpace(Var("OLLAMA_HOST"))
	scheme, hostport, ok := strings.Cut(s, "://")
	switch {
	case !ok:
		scheme, hostport = "http", s
	case scheme == "http":
		defaultPort = "80"
	case scheme == "https":
		defaultPort = "443"
	}

	hostport, path, _ := strings.Cut(hostport, "/")
	host, port, err := net.SplitHostPort(hostport)
	if err != nil {
		host, port = "127.0.0.1", defaultPort
		if ip := net.ParseIP(strings.Trim(hostport, "[]")); ip != nil {
			host = ip.String()
		} else if hostport != "" {
			host = hostport
		}
	}

	if n, err := strconv.ParseInt(port, 10, 32); err != nil || n > 65535 || n < 0 {
		slog.Warn("invalid port, using default", "port", port, "default", defaultPort)
		port = defaultPort
	}

	return &url.URL{
		Scheme: scheme,
		Host:   net.JoinHostPort(host, port),
		Path:   path,
	}
}

// Origins returns a list of allowed origins. Origins can be configured via the OLLAMA_ORIGINS environment variable.
func Origins() (origins []string) {
	if s := Var("OLLAMA_ORIGINS"); s != "" {
		origins = strings.Split(s, ",")
	}

	for _, origin := range []string{"localhost", "127.0.0.1", "0.0.0.0"} {
		origins = append(origins,
			fmt.Sprintf("http://%s", origin),
			fmt.Sprintf("https://%s", origin),
			fmt.Sprintf("http://%s", net.JoinHostPort(origin, "*")),
			fmt.Sprintf("https://%s", net.JoinHostPort(origin, "*")),
		)
	}

	origins = append(origins,
		"app://*",
		"file://*",
		"tauri://*",
	)

	return origins
}

// Models returns the path to the models directory. Models directory can be configured via the OLLAMA_MODELS environment variable.
// Default is $HOME/.ollama/models
func Models() string {
	if s := Var("OLLAMA_MODELS"); s != "" {
		return s
	}

	home, err := os.UserHomeDir()
	if err != nil {
		panic(err)
	}

	return filepath.Join(home, ".ollama", "models")
}

func Duration(k string, defaultValue time.Duration, zeroIsInfinite bool) func() time.Duration {
	return func() time.Duration {
		dur := defaultValue
		if s := Var(k); s != "" {
			if d, err := time.ParseDuration(s); err == nil {
				dur = d
			} else if n, err := strconv.ParseInt(s, 10, 64); err == nil {
				dur = time.Duration(n) * time.Second
			}
		}

		if dur < 0 || (dur == 0 && zeroIsInfinite) {
			return time.Duration(math.MaxInt64)
		}

		return dur
	}
}

var (
	// KeepAlive returns the duration that models stay loaded in memory. KeepAlive can be configured via the OLLAMA_KEEP_ALIVE environment variable.
	// Negative values are treated as infinite keep alive. Zero is treated as no keep alive.
	// Default is 5 minutes.
	KeepAlive = Duration("OLLAMA_KEEP_ALIVE", 5*time.Minute, false)

	// LoadTimeout returns the duration for stall detection during model loads. LoadTimeout can be configured via the OLLAMA_LOAD_TIMEOUT environment variable.
	// Negative or zero values are treated as infinite timeout.
	// Default is 5 minutes.
	LoadTimeout = Duration("OLLAMA_LOAD_TIMEOUT", 5*time.Minute, true)
)

func Bool(k string) func() bool {
	return func() bool {
		if s := Var(k); s != "" {
			b, err := strconv.ParseBool(s)
			if err != nil {
				return true
			}

			return b
		}

		return false
	}
}

var (
	// Debug enabled additional debug information.
	Debug = Bool("OLLAMA_DEBUG")
	// FlashAttention enables the experimental flash attention feature.
	FlashAttention = Bool("OLLAMA_FLASH_ATTENTION")
	// NoHistory disables readline history.
	NoHistory = Bool("OLLAMA_NOHISTORY")
	// NoPrune disables pruning of model blobs on startup.
	NoPrune = Bool("OLLAMA_NOPRUNE")
	// SchedSpread allows scheduling models across all GPUs.
	SchedSpread = Bool("OLLAMA_SCHED_SPREAD")
	// IntelGPU enables experimental Intel GPU detection.
	IntelGPU = Bool("OLLAMA_INTEL_GPU")
)

func String(s string) func() string {
	return func() string {
		return Var(s)
	}
}

var (
	LLMLibrary = String("OLLAMA_LLM_LIBRARY")
	TempDir    = String("OLLAMA_TMPDIR")

	CudaVisibleDevices    = String("CUDA_VISIBLE_DEVICES")
	HipVisibleDevices     = String("HIP_VISIBLE_DEVICES")
	RocrVisibleDevices    = String("ROCR_VISIBLE_DEVICES")
	GpuDeviceOrdinal      = String("GPU_DEVICE_ORDINAL")
	HsaOverrideGfxVersion = String("HSA_OVERRIDE_GFX_VERSION")
)


func Uint[T uint | uint16 | uint32 | uint64](key string, defaultValue T) func() T {
	return func() T {
		if s := Var(key); s != "" {
			if n, err := strconv.ParseUint(s, 10, 64); err != nil {
				slog.Warn("invalid environment variable, using default", "key", key, "value", s, "default", defaultValue)
			} else {
				return T(n)
			}
		}

		return defaultValue
	}
}

var (
	// NumParallel sets the number of parallel model requests. NumParallel can be configured via the OLLAMA_NUM_PARALLEL environment variable.
	NumParallel = Uint("OLLAMA_NUM_PARALLEL", uint(0))
	// MaxRunners sets the maximum number of loaded models. MaxRunners can be configured via the OLLAMA_MAX_LOADED_MODELS environment variable.
	MaxRunners = Uint("OLLAMA_MAX_LOADED_MODELS", uint(0))
	// MaxQueue sets the maximum number of queued requests. MaxQueue can be configured via the OLLAMA_MAX_QUEUE environment variable.
	MaxQueue = Uint("OLLAMA_MAX_QUEUE", uint(512))
	// MaxVRAM sets a maximum VRAM override in bytes. MaxVRAM can be configured via the OLLAMA_MAX_VRAM environment variable.
	MaxVRAM = Uint("OLLAMA_MAX_VRAM", uint(0))
	// GPUOverhead reserves a portion of VRAM per GPU. GPUOverhead can be configured via the OLLAMA_GPU_OVERHEAD environment variable.
	GPUOverhead = Uint("OLLAMA_GPU_OVERHEAD", uint64(0))
)

type desc struct {
	name         string
	usage        string
	value        any
	defaultValue any
}

func (e desc) String() string {
	return fmt.Sprintf("%s:%v", e.name, e.value)
}

func Vars() []desc {
	s := []desc{
		{"OLLAMA_DEBUG", "Enable debug", Debug(), false},
		{"OLLAMA_FLASH_ATTENTION", "Enabled flash attention", FlashAttention(), false},
		{"OLLAMA_GPU_OVERHEAD", "Reserve a portion of VRAM per GPU", GPUOverhead(), 0},
		{"OLLAMA_HOST", "Listen address and port", Host(), "127.0.0.1:11434"},
		{"OLLAMA_KEEP_ALIVE", "Duration of inactivity before models are unloaded", KeepAlive(), 5 * time.Minute},
		{"OLLAMA_LLM_LIBRARY", "Set LLM library to bypass autodetection", LLMLibrary(), nil},
		{"OLLAMA_LOAD_TIMEOUT", "Duration for stall detection during model loads", LoadTimeout(), 5 * time.Minute},
		{"OLLAMA_MAX_LOADED_MODELS", "Maximum number of loaded models per GPU", MaxRunners(), nil},
		{"OLLAMA_MAX_QUEUE", "Maximum number of queued requests", MaxQueue(), nil},
		{"OLLAMA_MAX_VRAM", "Maximum VRAM to consider for model offloading", MaxVRAM(), nil},
		{"OLLAMA_MODELS", "Path override for models directory", Models(), nil},
		{"OLLAMA_NOHISTORY", "Disable readline history", NoHistory(), false},
		{"OLLAMA_NOPRUNE", "Disable unused blob pruning", NoPrune(), false},
		{"OLLAMA_NUM_PARALLEL", "Maximum number of parallel requests before requests are queued", NumParallel(), nil},
		{"OLLAMA_ORIGINS", "Additional HTTP Origins to allow", Origins(), nil},
		{"OLLAMA_SCHED_SPREAD", "Always schedule model across all GPUs", SchedSpread(), false},
		{"OLLAMA_TMPDIR", "Path override for temporary directory", TempDir(), nil},

		// informational
		{"HTTPS_PROXY", "Proxy for HTTPS requests", os.Getenv("HTTPS_PROXY"), nil},
		{"HTTP_PROXY", "Proxy for HTTP requests", os.Getenv("HTTP_PROXY"), nil},
		{"NO_PROXY", "No proxy for these hosts", os.Getenv("NO_PROXY"), nil},
	}

	if runtime.GOOS != "windows" {
		s = append(
			s,
			desc{"https_proxy", "Proxy for HTTPS requests", os.Getenv("https_proxy"), nil},
			desc{"http_proxy", "Proxy for HTTP requests", os.Getenv("http_proxy"), nil},
			desc{"no_proxy", "No proxy for these hosts", os.Getenv("no_proxy"), nil},
		)
	}

	if runtime.GOOS != "darwin" {
		s = append(
			s,
			desc{"CUDA_VISIBLE_DEVICES", "Set which NVIDIA devices are visible", CudaVisibleDevices(), nil},
			desc{"HIP_VISIBLE_DEVICES", "Set which AMD devices are visible", HipVisibleDevices(), nil},
			desc{"ROCR_VISIBLE_DEVICES", "Set which AMD devices are visible", RocrVisibleDevices(), nil},
			desc{"GPU_DEVICE_ORDINAL", "Set which AMD devices are visible", GpuDeviceOrdinal(), nil},
			desc{"HSA_OVERRIDE_GFX_VERSION", "Override the gfx used for all detected AMD GPUs", HsaOverrideGfxVersion(), nil},
			desc{"OLLAMA_INTEL_GPU", "Enable experimental Intel GPU detection", IntelGPU(), nil},
		)
	}

	return s
}

func Describe(s ...string) map[string]string {
	vars := Vars()
	m := make(map[string]string, len(s))
	for _, k := range s {
		if i := slices.IndexFunc(vars, func(e desc) bool { return e.name == k }); i != -1 {
			m[k] = vars[i].usage
			if vars[i].defaultValue != nil {
				m[k] = fmt.Sprintf("%s (default: %v)", vars[i].usage, vars[i].defaultValue)
			}
		}
	}

	return m
}

// Var returns an environment variable stripped of leading and trailing quotes or spaces
func Var(key string) string {
	return strings.Trim(strings.TrimSpace(os.Getenv(key)), "\"'")
}

// On windows, we keep the binary at the top directory, but
// other platforms use a "bin" directory, so this returns ".."
func LibRelativeToExe() string {
	if runtime.GOOS == "windows" {
		return "."
	}
	return ".."
}
