package provider

import (
	"errors"
	"io"
	"net"
	"testing"

	"github.com/anthropics/anthropic-sdk-go"
	openai "github.com/openai/openai-go/v3"
)

// TestIsRetryableStatusCodes covers the SDK API-error path: both providers'
// *Error types expose StatusCode, and 429/5xx are retryable while 4xx (other
// than 429) and 200 are not. We construct *Error directly with only StatusCode
// set; isRetryable reads only that field and never calls Error() (which would
// require a non-nil Request), so this is safe.
func TestIsRetryableStatusCodes(t *testing.T) {
	cases := []struct {
		name    string
		err     error
		retryab bool
	}{
		{"anthropic 429", &anthropic.Error{StatusCode: 429}, true},
		{"anthropic 500", &anthropic.Error{StatusCode: 500}, true},
		{"anthropic 503", &anthropic.Error{StatusCode: 503}, true},
		{"anthropic 400", &anthropic.Error{StatusCode: 400}, false},
		{"anthropic 401", &anthropic.Error{StatusCode: 401}, false},
		{"anthropic 404", &anthropic.Error{StatusCode: 404}, false},
		{"openai 429", &openai.Error{StatusCode: 429}, true},
		{"openai 502", &openai.Error{StatusCode: 502}, true},
		{"openai 400", &openai.Error{StatusCode: 400}, false},
		{"nil", nil, false},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := isRetryable(c.err); got != c.retryab {
				t.Fatalf("isRetryable(%v) = %v, want %v", c.name, got, c.retryab)
			}
		})
	}
}

// TestIsRetryableNetwork covers the network-layer path: timeouts and EOF are
// retryable; a wrapped timeout is still detected via errors.As.
func TestIsRetryableNetwork(t *testing.T) {
	// io.EOF / io.ErrUnexpectedEOF (gateway mid-stream close).
	if !isRetryable(io.EOF) {
		t.Fatal("io.EOF should be retryable")
	}
	if !isRetryable(io.ErrUnexpectedEOF) {
		t.Fatal("io.ErrUnexpectedEOF should be retryable")
	}

	// A net.Error reporting a timeout.
	ne := &timeoutError{}
	if !isRetryable(ne) {
		t.Fatal("timeout net.Error should be retryable")
	}

	// A plain non-timeout error (e.g. a generic sentinel) is not retryable.
	plain := errors.New("some non-transient failure")
	if isRetryable(plain) {
		t.Fatal("plain non-transient error should not be retryable")
	}

	// A wrapped timeout is still detected via errors.As (errors.Join produces a
	// multi-error that errors.As unwraps).
	if !isRetryable(errors.Join(ne, plain)) {
		t.Fatal("wrapped timeout should be retryable via errors.As")
	}
}

// timeoutError is a minimal net.Error that reports a timeout, for testing the
// net.Error.Timeout() branch of isRetryable without a real network operation.
type timeoutError struct{}

func (timeoutError) Error() string   { return "i/o timeout" }
func (timeoutError) Timeout() bool   { return true }
func (timeoutError) Temporary() bool { return true }

var _ net.Error = timeoutError{}
