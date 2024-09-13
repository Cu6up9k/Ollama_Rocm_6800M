package telemetry

import (
	"context"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/exporters/prometheus"
	"go.opentelemetry.io/otel/metric"
	metricsdk "go.opentelemetry.io/otel/sdk/metric"

)

const (
    namespace = "ollama"
)

var (
    meter = otel.Meter(namespace)
)

type Metrics struct {
    Request metric.Int64Counter
    ModelActions metric.Int64Counter
    Chat metric.Int64Counter
}

func NewMetrics() *Metrics {
	req, err := meter.Int64Counter(
		"requests_total",
		metric.WithDescription("The total number of requests on all endpoints."),
		metric.WithUnit("requests"),
	)
	if err != nil {
        return &Metrics{}
    }
	modelActions, err := meter.Int64Counter(
		"model_actions_total",
		metric.WithDescription("The total number of model actions that have been attempted."),
		metric.WithUnit("requests"),
	)
	if err != nil {
        return &Metrics{}
    }
	chat, err := meter.Int64Counter(
		"chat_requests_total",
		metric.WithDescription("The total number of requests that have been attempted on chat endpoint."),
		metric.WithUnit("requests"),
	)
	if err != nil {
        return &Metrics{}
    }


    return &Metrics{
        Request: req,
        ModelActions: modelActions,
        Chat: chat,
    }
}

func (m *Metrics) RecordRequest(ctx context.Context, action string, statusCode int64, status string) {
    m.Request.Add(ctx, 1, metric.WithAttributes(
        attribute.String("action", action),
        attribute.Int64("status_code", statusCode),
        attribute.String("status", status),
    ))
}

func (m *Metrics) RecordModel(ctx context.Context, action string, statusCode int64, status string) {
    m.ModelActions.Add(ctx, 1, metric.WithAttributes(
        attribute.String("action", action),
        attribute.Int64("status_code", statusCode),
        attribute.String("status", status),
    ))
}

func (m *Metrics) RecordChat(ctx context.Context, action string, statusCode int64, status string) {
    m.Chat.Add(ctx, 1, metric.WithAttributes(
        attribute.String("action", action),
        attribute.Int64("status_code", statusCode),
        attribute.String("status", status),
    ))
}


func NewPrometheusMeterProvider() (*metricsdk.MeterProvider, error) {
    // Create a Prometheus exporter instance
    meterExporter, err := prometheus.New()
    if err != nil {
        return nil, err
    }
    meterProvider := metricsdk.NewMeterProvider(metricsdk.WithReader(meterExporter))
	// meter := meterProvider.Meter("ollama")7

    return meterProvider,nil
}