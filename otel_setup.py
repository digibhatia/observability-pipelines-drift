"""
Wires up the OpenTelemetry SDK for BOTH signals Satark needs, where Nidan
only needed one:

  - traces  - unchanged in spirit from Nidan, but now exported to TWO
              backends at once when Langfuse is enabled: Jaeger (via the
              Collector, for the trace-shape debugging Nidan taught) and
              Langfuse (via the Collector's second OTLP exporter, for the
              GenAI-aware trace views - cost-by-model, latency-by-tool - a
              general-purpose backend like Jaeger does not give you out of
              the box). Nothing in app.py changes to get this: it is a
              Collector pipeline fan-out, configured once, in
              otel-collector-config.yaml.

  - metrics - NEW. A MeterProvider, exported over OTLP to the same
              Collector, which has its own pipeline turning them into a
              Prometheus-scrapeable endpoint. This is what backs the M4
              Grafana dashboard: metrics.py defines the actual instruments
              (counters, histograms), this file only decides WHERE they go.

Swap OTEL_EXPORTER in .env to "console" to print both signals to stdout with
no Collector running at all - useful for a first look before standing up
the full stack.
"""
import os

from opentelemetry import trace, metrics
from opentelemetry.sdk.resources import Resource, SERVICE_NAME, SERVICE_VERSION
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.trace.sampling import TraceIdRatioBased, ParentBased
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import (
    PeriodicExportingMetricReader, ConsoleMetricExporter,
)

_initialized = False


def _build_resource() -> Resource:
    return Resource.create({
        SERVICE_NAME: "satark-lloyds-agent",
        SERVICE_VERSION: "0.1.0",
        "deployment.environment": os.environ.get("DEPLOY_ENV", "lab"),
    })


def init_tracing():
    global _initialized
    if _initialized:
        return trace.get_tracer_provider(), metrics.get_meter_provider()

    resource = _build_resource()
    sample_ratio = float(os.environ.get("OTEL_TRACE_SAMPLE_RATIO", "1.0"))
    sampler = ParentBased(TraceIdRatioBased(sample_ratio))
    trace_provider = TracerProvider(resource=resource, sampler=sampler)

    exporter_kind = os.environ.get("OTEL_EXPORTER", "otlp").lower()
    if exporter_kind == "console":
        span_exporter = ConsoleSpanExporter()
        metric_reader = PeriodicExportingMetricReader(ConsoleMetricExporter())
    else:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
        endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
        span_exporter = OTLPSpanExporter(endpoint=endpoint, insecure=True)
        metric_reader = PeriodicExportingMetricReader(
            OTLPMetricExporter(endpoint=endpoint, insecure=True),
            export_interval_millis=int(os.environ.get("OTEL_METRIC_EXPORT_INTERVAL_MS", "5000")),
        )

    trace_provider.add_span_processor(BatchSpanProcessor(span_exporter))
    trace.set_tracer_provider(trace_provider)

    meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
    metrics.set_meter_provider(meter_provider)

    _initialized = True
    return trace_provider, meter_provider


def get_tracer(name: str = "satark"):
    init_tracing()
    return trace.get_tracer(name)


def get_meter(name: str = "satark"):
    init_tracing()
    return metrics.get_meter(name)
