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

(ert-deftest llm-setup-test-model-registry-sorted-and-unique ()
  "Keep model family names sorted case-insensitively and unique."
  (let ((keys
         (mapcar
          (lambda (model)
            (downcase (symbol-name (llm-setup-model-name model))))
          llm-setup-models-list)))
    (should (equal keys (sort (copy-sequence keys) #'string<)))
    (should
     (= (length keys)
        (length (delete-dups (copy-sequence keys)))))))

(ert-deftest llm-setup-test-resident-model-references ()
  "Require resident and preloaded names to resolve to declared instances."
  (should
   (equal llm-setup-llama-swap-always-on-models
          '(Qwen3.6-27B-Instruct GLM-5.2 bge-m3)))
  (should
   (equal llm-setup-llama-swap-preload-models
          '(Qwen3.6-27B-Instruct bge-m3)))
  (let ((instance-names
         (mapcar
          (lambda (entry)
            (llm-setup-get-instance-name (car entry) (cdr entry)))
          (llm-setup-instances-list))))
    (dolist (name (append llm-setup-llama-swap-always-on-models
                          llm-setup-llama-swap-preload-models))
      (should (memq name instance-names)))
    (dolist (name llm-setup-llama-swap-preload-models)
      (should (memq name llm-setup-llama-swap-always-on-models)))))

(ert-deftest llm-setup-test-resident-model-host-eligibility ()
  "Generate resident groups and preload hooks from host-eligible models."
  (let ((eligible
         (lambda (hostname)
           (cl-remove-if-not
            (lambda (name)
              (seq-some
               (lambda (entry)
                 (let ((model (car entry))
                       (instance (cdr entry)))
                   (and
                    (eq name
                        (llm-setup-get-instance-name model instance))
                    (memq (llm-setup-instance-provider instance)
                          '(local vibe-proxy))
                    (member hostname
                            (llm-setup-instance-hostnames instance)))))
               (llm-setup-instances-list)))
            llm-setup-llama-swap-always-on-models))))
    (let ((hera (funcall eligible "hera"))
          (clio (funcall eligible "clio"))
          (hooks
           "\nhooks:\n  on_startup:\n    preload:\n      - Qwen3.6-27B-Instruct\n      - bge-m3\n"))
      (should (equal hera '(Qwen3.6-27B-Instruct GLM-5.2 bge-m3)))
      (should (equal clio '(Qwen3.6-27B-Instruct bge-m3)))
      (should
       (equal
        (llm-setup--generate-llama-swap-groups hera)
        "\ngroups:\n  always_on:\n    swap: false\n    exclusive: false\n    members:\n      - Qwen3.6-27B-Instruct\n      - GLM-5.2\n      - bge-m3\n"))
      (should
       (equal
        (llm-setup--generate-llama-swap-groups clio)
        "\ngroups:\n  always_on:\n    swap: false\n    exclusive: false\n    members:\n      - Qwen3.6-27B-Instruct\n      - bge-m3\n"))
      (should (equal (llm-setup--generate-llama-swap-hooks hera) hooks))
      (should (equal (llm-setup--generate-llama-swap-hooks clio) hooks)))))

(provide 'llm-setup-test)

;;; llm-setup-test.el ends here
