package main

import (
	"net/http/httptest"
	"testing"
)

func TestIsLoopbackAddr(t *testing.T) {
	cases := map[string]bool{
		"127.0.0.1:8080": true, "localhost:8080": true, "[::1]:8080": true,
		":8080": false, "0.0.0.0:8080": false, "192.168.1.5:8080": false,
	}
	for addr, want := range cases {
		if got := isLoopbackAddr(addr); got != want {
			t.Errorf("isLoopbackAddr(%q) = %v, want %v", addr, got, want)
		}
	}
}

func TestOriginAndHost(t *testing.T) {
	loopbackOnly = true
	cases := []struct {
		host, origin string
		wantHost     bool
		wantOrigin   bool
	}{
		{"localhost:8080", "http://localhost:8080", true, true},
		{"127.0.0.1:8080", "http://127.0.0.1:8080", true, true},
		{"127.0.0.1:8080", "", true, true},                             // non-browser client
		{"127.0.0.1:8080", "https://evil.example", true, false},        // cross-site page
		{"127.0.0.1:8080", "http://localhost:9999", true, false},       // other local port
		{"127.0.0.1:8080", "null", true, false},                        // sandboxed iframe / file://
		{"evil.example:8080", "http://evil.example:8080", false, true}, // DNS rebinding
	}
	for _, c := range cases {
		r := httptest.NewRequest("GET", "/ws", nil)
		r.Host = c.host
		if c.origin != "" {
			r.Header.Set("Origin", c.origin)
		}
		if got := hostAllowed(r); got != c.wantHost {
			t.Errorf("hostAllowed(host %q) = %v, want %v", c.host, got, c.wantHost)
		}
		if got := originAllowed(r); got != c.wantOrigin {
			t.Errorf("originAllowed(host %q, origin %q) = %v, want %v", c.host, c.origin, got, c.wantOrigin)
		}
	}
}
