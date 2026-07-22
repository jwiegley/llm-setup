;;; llm-setup-test.el --- Tests for llm-setup  -*- lexical-binding: t; -*-

;;; Commentary:

;; Regression tests for generated model configuration.

;;; Code:

(require 'ert)
(require 'llm-setup)

(declare-function llm-setup-aider-model-name "llm-setup")
(declare-function llm-setup--instance-eligible-for-host-p "llm-setup")

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

(ert-deftest llm-setup-test-openrouter-routes ()
  "Generate exact GPTel and LiteLLM routes for OpenRouter models."
  (dolist (case '((Kimi-K3
                   moonshotai/kimi-k3
                   openrouter/moonshotai/kimi-k3
                   1048576)
                  (Qwen3.7-Max
                   qwen/qwen3.7-max
                   openrouter/qwen/qwen3.7-max
                   1000000)))
    (pcase-let
        ((`(,family-name ,instance-name ,route-name ,context-length) case))
      (let* ((model
              (seq-find
               (lambda (candidate)
                 (eq (llm-setup-model-name candidate) family-name))
               llm-setup-models-list))
             (instances (and model (llm-setup-model-instances model)))
             (instance (car instances)))
        (should model)
        (should (= 1 (length instances)))
        (should (eq (llm-setup-instance-name instance) instance-name))
        (should (eq (llm-setup-instance-provider instance) 'openrouter))
        (should (= context-length (llm-setup-model-context-length model)))
        (should (llm-setup-model-supports-reasoning model))
        (should
         (equal
          (mapcar
           #'car
           (llm-setup-get-instance-gptel-backend model instance))
          (list route-name)))
        (with-temp-buffer
          (llm-setup-insert-instance-litellm model instance)
          (let ((yaml (buffer-string)))
            (should
             (string-match-p
              (regexp-quote
               (format "\n  - model_name: %s\n" route-name))
              yaml))
            (should
             (string-match-p
              (regexp-quote (format "\n      model: %s\n" route-name))
              yaml))
            (should
             (string-match-p
              (regexp-quote
               "\n      litellm_credential_name: openrouter_credential\n")
              yaml))
            (should
             (string-match-p
              (regexp-quote "\n      supports_reasoning: true\n")
              yaml))))))))

(ert-deftest llm-setup-test-openrouter-glm-family ()
  "Keep local and OpenRouter GLM-5.2 instances in one family."
  (let ((models
         (seq-filter
          (lambda (model)
            (eq (llm-setup-model-name model) 'GLM-5.2))
          llm-setup-models-list)))
    (should (= 1 (length models)))
    (let* ((model (car models))
           (instances (llm-setup-model-instances model))
           (local
            (seq-find
             (lambda (instance)
               (eq (llm-setup-instance-provider instance) 'local))
             instances))
           (openrouter
            (seq-find
             (lambda (instance)
               (eq (llm-setup-instance-provider instance) 'openrouter))
             instances))
           (route-name 'openrouter/z-ai/glm-5.2))
      (should (= 2 (length instances)))
      (should local)
      (should openrouter)
      (should (= 200000 (llm-setup-model-context-length model)))
      (should (= 1.0 (llm-setup-model-temperature model)))
      (should
       (equal (llm-setup-instance-model-path local)
              "~/Models/unsloth_GLM-5.2-GGUF"))
      (should (= 200000
                 (llm-setup-get-instance-context-length model local)))
      (should (eq (llm-setup-instance-name openrouter) 'z-ai/glm-5.2))
      (should (= 1048576
                 (llm-setup-instance-context-length openrouter)))
      (should (= 1048576
                 (llm-setup-get-instance-context-length
                  model openrouter)))
      (should
       (equal
        (mapcar
         #'car
         (llm-setup-get-instance-gptel-backend model openrouter))
        (list route-name)))
      (with-temp-buffer
        (llm-setup-insert-instance-litellm model openrouter)
        (let ((yaml (buffer-string)))
          (should
           (string-match-p
            (regexp-quote
             (format "\n  - model_name: %s\n" route-name))
            yaml))
          (should
           (string-match-p
            (regexp-quote (format "\n      model: %s\n" route-name))
            yaml)))))))

(ert-deftest llm-setup-test-resident-model-references ()
  "Require resident and preloaded names to resolve to declared instances."
  (should
   (equal llm-setup-llama-swap-always-on-models
          '(GLM-5.2 bge-m3)))
  (should
   (equal llm-setup-llama-swap-preload-models
          '(bge-m3)))
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
  "Use the generator's host predicate to select resident candidates."
  (cl-labels
      ((resident-candidates
         (hostname)
         (let (names)
           (dolist (entry (llm-setup-instances-list))
             (let* ((model (car entry))
                    (instance (cdr entry))
                    (name (llm-setup-get-instance-name model instance)))
               (when
                   (and
                    (llm-setup--instance-eligible-for-host-p
                     instance hostname)
                    (memq name llm-setup-llama-swap-always-on-models))
                 (cl-pushnew name names))))
           (sort
            names
            (lambda (left right)
              (string< (downcase (symbol-name left))
                       (downcase (symbol-name right))))))))
    (should
     (equal (resident-candidates "hera")
            '(bge-m3 GLM-5.2)))
    (should
     (equal (resident-candidates "clio")
            '(bge-m3)))))

(ert-deftest llm-setup-test-llama-swap-policy-rendering ()
  "Render resident, exclusive, and preload sections from emitted models."
  (let ((emitted
         '(bge-m3 GLM-5.2 Huihui-Qwable-3.6-27b-abliterated-MTP)))
    (should
     (equal
      (llm-setup--generate-llama-swap-groups emitted)
      "\ngroups:\n  always_on:\n    swap: false\n    exclusive: false\n    members:\n      - bge-m3\n      - GLM-5.2\n  exclusive_models:\n    swap: true\n    exclusive: false\n    members:\n      - Huihui-Qwable-3.6-27b-abliterated-MTP\n"))
    (should
     (equal
      (llm-setup--generate-llama-swap-hooks emitted)
      "\nhooks:\n  on_startup:\n    preload:\n      - bge-m3\n"))))

(ert-deftest llm-setup-test-llama-swap-concurrency-limit ()
  "Emit a per-model concurrency limit only when one is configured."
  (let ((model (make-llm-setup-model :name 'test-model :context-length 1))
        (instance
         (make-llm-setup-instance
          :file-path "/tmp/test-model.gguf"
          :concurrency-limit 32))
        (llm-setup-llama-server-executable "true"))
    (with-temp-buffer
      (llm-setup-insert-instance-llama-swap model instance "hera")
      (should
       (string-match-p
        (regexp-quote
         "\n    concurrencyLimit: 32\n    checkEndpoint: /health\n")
        (buffer-string))))
    (setf (llm-setup-instance-concurrency-limit instance) nil)
    (with-temp-buffer
      (llm-setup-insert-instance-llama-swap model instance "hera")
      (should-not (string-match-p "concurrencyLimit" (buffer-string))))))

(ert-deftest llm-setup-test-default-gptel-model-exists ()
  "Require the shared client default to resolve to one GPTel backend model."
  (should
   (eq llm-setup-default-instance-name
       'hera/omlx/Qwen3.6-27B-oQ4e-mtp))
  (should
   (equal (llm-setup-aider-model-name)
          "openai/hera/omlx/Qwen3.6-27B-oQ4e-mtp"))
  (cl-letf (((symbol-function 'yaml-mode) #'fundamental-mode))
    (with-current-buffer (llm-setup-generate-promptdeploy-yaml)
      (should
       (string-prefix-p
        "defaults:\n  provider: litellm\n  model: hera/omlx/Qwen3.6-27B-oQ4e-mtp\n\nproviders:\n"
        (buffer-string)))))
  (should
   (= 1
      (cl-count
       llm-setup-default-instance-name
       (mapcar #'car (llm-setup-gptel-backends))
       :test #'eq))))

(provide 'llm-setup-test)

;;; llm-setup-test.el ends here
