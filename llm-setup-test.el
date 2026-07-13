;;; llm-setup-test.el --- Tests for llm-setup  -*- lexical-binding: t; -*-

;;; Commentary:

;; Regression tests for generated model configuration.

;;; Code:

(require 'ert)
(require 'llm-setup)

(ert-deftest llm-setup-test-litellm-provider-model-alias ()
  "Keep the public route distinct from its provider-facing model name."
  (let ((model (make-llm-setup-model :name 'claude-fable))
        (instance
         (make-llm-setup-instance
          :name 'claude-fable-5-thinking-32000
          :model-name 'claude-fable-5
          :provider 'vibe-proxy
          :hostnames '("hera"))))
    (with-temp-buffer
      (llm-setup-insert-instance-litellm model instance)
      (let ((yaml (buffer-string)))
        (should
         (string-match-p
          (regexp-quote
           "model_name: hera/claude-fable-5-thinking-32000\n")
          yaml))
        (should
         (string-match-p
          (regexp-quote "model: openai/claude-fable-5\n")
          yaml))
        (should-not
         (string-match-p
          (regexp-quote
           "model: openai/claude-fable-5-thinking-32000\n")
          yaml))))))

(ert-deftest llm-setup-test-litellm-provider-model-name-fallback ()
  "Use the public instance name when no provider override is configured."
  (let ((model (make-llm-setup-model :name 'claude-fable))
        (instance
         (make-llm-setup-instance
          :name 'claude-fable-5
          :provider 'anthropic
          :hostnames '("hera"))))
    (with-temp-buffer
      (llm-setup-insert-instance-litellm model instance)
      (let ((yaml (buffer-string)))
        (should
         (string-match-p
          (regexp-quote "model_name: anthropic/claude-fable-5\n")
          yaml))
        (should
         (string-match-p
          (regexp-quote "model: anthropic/claude-fable-5\n")
          yaml))))))

(ert-deftest llm-setup-test-claude-fable-routes ()
  "Generate the provider-facing Fable name for every public route."
  (let ((model
         (seq-find
          (lambda (candidate)
            (eq (llm-setup-model-name candidate) 'claude-fable))
          llm-setup-models-list)))
    (should model)
    (let ((instances (llm-setup-model-instances model)))
      (should (= 4 (length instances)))
      (dolist (instance instances)
        (should
         (eq (llm-setup-get-instance-model-name model instance)
             'claude-fable-5))
        (with-temp-buffer
          (llm-setup-insert-instance-litellm model instance)
          (let ((yaml (buffer-string)))
            (should
             (string-match-p
              "^      model: \\(?:openai\\|anthropic\\)/claude-fable-5$"
              yaml))
            (should-not
             (string-match-p
              "^      model: [^\n]*/claude-fable$"
              yaml))))))))

(provide 'llm-setup-test)

;;; llm-setup-test.el ends here
