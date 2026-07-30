;;; llm-setup-test.el --- Tests for llm-setup  -*- lexical-binding: t; -*-

;;; Commentary:

;; Regression tests for generated model configuration.

;;; Code:

(require 'ert)
(require 'llm-setup)

(declare-function llm-setup-aider-model-name "llm-setup")
(declare-function llm-setup--instance-eligible-for-host-p "llm-setup")

(defun llm-setup-test--nix-model-registry ()
  "Return the rendered Nix model registry parsed as alists and lists."
  (json-parse-string
   (llm-setup-render-nix-model-registry)
   :object-type 'alist
   :array-type 'list
   :null-object nil
   :false-object :json-false))

(defun llm-setup-test--nix-provider (registry id)
  "Return the provider with ID from REGISTRY."
  (seq-find
   (lambda (provider)
     (equal (alist-get 'id provider) id))
   (alist-get 'providers registry)))

(defun llm-setup-test--nix-model (registry provider id)
  "Return PROVIDER and ID's model from REGISTRY."
  (seq-find
   (lambda (model)
     (and (equal (alist-get 'provider model) provider)
          (equal (alist-get 'id model) id)))
   (alist-get 'models registry)))

(defun llm-setup-test--keys-follow-p (object allowed)
  "Return non-nil when OBJECT's keys are the ordered subset of ALLOWED."
  (let ((keys (mapcar #'car object)))
    (equal keys
           (seq-filter (lambda (key) (memq key keys)) allowed))))

(defun llm-setup-test--assert-hosts (object)
  "Assert that OBJECT's optional hosts field is valid when present."
  (when-let* ((hosts-cell (assq 'hosts object)))
    (let ((hosts (cdr hosts-cell)))
      (should (consp hosts))
      (should (seq-every-p #'stringp hosts))
      (should (seq-every-p (lambda (host) (member host '("hera" "clio")))
                           hosts))
      (should
       (= (length hosts)
          (length (delete-dups (copy-sequence hosts))))))))


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
  "Generate exact GPTel routes for OpenRouter models."
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
          (list route-name)))))))

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
                 (llm-setup-get-instance-context-length model openrouter)))
      (should
       (equal
        (mapcar
         #'car
         (llm-setup-get-instance-gptel-backend model openrouter))
        (list route-name))))))

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

(ert-deftest llm-setup-test-llama-swap-generates-per-host ()
  "Generate only host-eligible llama-swap entries for hera and clio."
  (let ((llm-setup-models-list
         (list
          (make-llm-setup-model
           :name 'both
           :instances
           (list (make-llm-setup-instance :hostnames '("hera" "clio"))))
          (make-llm-setup-model
           :name 'hera-only
           :instances
           (list (make-llm-setup-instance :hostnames '("hera"))))
          (make-llm-setup-model
           :name 'clio-only
           :instances
           (list (make-llm-setup-instance :hostnames '("clio"))))
          (make-llm-setup-model
           :name 'remote
           :instances
           (list (make-llm-setup-instance :provider 'openrouter))))))
    (cl-letf (((symbol-function 'llm-setup-insert-instance-llama-swap)
               (lambda (model _instance _hostname &optional _cache)
                 (insert (format "\n    %s" (llm-setup-model-name model)))))
              ((symbol-function 'yaml-mode) #'ignore))
      (let ((hera
             (with-current-buffer
                 (llm-setup-generate-llama-swap-yaml "hera")
               (buffer-string)))
            (clio
             (with-current-buffer
                 (llm-setup-generate-llama-swap-yaml "clio")
               (buffer-string))))
        (dolist (name '("both" "hera-only"))
          (should (string-match-p (regexp-quote name) hera)))
        (dolist (name '("both" "clio-only"))
          (should (string-match-p (regexp-quote name) clio)))
        (should-not (string-match-p "clio-only" hera))
        (should-not (string-match-p "hera-only" clio))
        (should-not (string-match-p "remote" hera))
        (should-not (string-match-p "remote" clio))))))

(ert-deftest llm-setup-test-build-llama-swap-targets-hera-and-clio ()
  "Write llama-swap YAML to ~/Models on hera and clio, then stop each service."
  (let ((llm-setup-gguf-models "/Users/johnw/Models")
        writes
        calls)
    (cl-letf (((symbol-function 'llm-setup-generate-llama-swap-yaml)
               (lambda (hostname)
                 (with-current-buffer (get-buffer-create " *llm-setup-test-yaml*")
                   (erase-buffer)
                   (insert hostname)
                   (current-buffer))))
              ((symbol-function 'write-file)
               (lambda (path &rest _) (push path writes)))
              ((symbol-function 'call-process)
               (lambda (&rest args) (push args calls))))
      (unwind-protect
          (progn
            (llm-setup-build-llama-swap-yaml)
            (llm-setup-build-llama-swap-yaml "clio")
            (should
             (equal
              (nreverse writes)
              '("/Users/johnw/Models/llama-swap.yaml"
                "/ssh:clio:/Users/johnw/Models/llama-swap.yaml")))
            (should
             (equal
              (nreverse calls)
              '(("killall" nil nil nil "llama-swap")
                ("ssh" nil nil nil "clio" "killall" "llama-swap")))))
        (kill-buffer " *llm-setup-test-yaml*")))))

(ert-deftest llm-setup-test-default-gptel-model-exists ()
  "Require the shared client default to resolve to one GPTel backend model."
  (should
   (eq llm-setup-default-instance-name
       'hera/omlx/Qwen3.6-27B-oQ4e-mtp))
  (should
   (equal (llm-setup-aider-model-name)
          "openai/hera/omlx/Qwen3.6-27B-oQ4e-mtp"))
  (should
   (= 1
      (cl-count
       llm-setup-default-instance-name
       (mapcar #'car (llm-setup-gptel-backends))
       :test #'eq))))

(ert-deftest llm-setup-test-nix-model-registry-exact-schema ()
  "Render only the locked top-level, provider, and model fields."
  (let* ((registry (llm-setup-test--nix-model-registry))
         (providers (alist-get 'providers registry))
         (models (alist-get 'models registry)))
    (should
     (equal (mapcar #'car registry)
            '(schemaVersion selections providers models)))
    (should (= 2 (alist-get 'schemaVersion registry)))
    (let ((selections (alist-get 'selections registry)))
      (should
       (equal (mapcar #'car selections)
              '(default claudeDefault claudeHaiku claudeSubagent)))
      (dolist (selection selections)
        (should (equal (mapcar #'car (cdr selection)) '(provider model)))
        (dolist (key '(provider model))
          (let ((value (alist-get key (cdr selection))))
            (should (stringp value))
            (should-not (string-empty-p value))))))
    (dolist (provider providers)
      (should
       (llm-setup-test--keys-follow-p
        provider '(id displayName baseUrl apiKey hosts)))
      (should
       (equal (seq-take (mapcar #'car provider) 4)
              '(id displayName baseUrl apiKey)))
      (dolist (key '(id displayName baseUrl))
        (let ((value (alist-get key provider)))
          (should (stringp value))
          (should-not (string-empty-p value))))
      (let* ((api-key (alist-get 'apiKey provider))
             (key (caar api-key))
             (value (cdar api-key)))
        (should (= 1 (length api-key)))
        (should (stringp value))
        (should-not (string-empty-p value))
        (pcase key
          ('env
           (should (string-match-p "\\`[A-Z][A-Z0-9_]*\\'" value)))
          ('nonSecret
           (should
            (member value '("dummy-api-key" "dummy-key" "not-needed"))))
          (_ (ert-fail (format "Unexpected apiKey shape: %S" api-key)))))
      (llm-setup-test--assert-hosts provider))
    (dolist (model models)
      (should
       (llm-setup-test--keys-follow-p
        model
        '(provider id displayName maxOutputTokens contextLimit outputLimit
                   hosts)))
      (should
       (equal (seq-take (mapcar #'car model) 4)
              '(provider id displayName maxOutputTokens)))
      (dolist (key '(provider id displayName))
        (let ((value (alist-get key model)))
          (should (stringp value))
          (should-not (string-empty-p value))))
      (dolist (key '(maxOutputTokens contextLimit outputLimit))
        (when-let* ((cell (assq key model)))
          (should (integerp (cdr cell)))
          (should (> (cdr cell) 0))))
      (llm-setup-test--assert-hosts model))))

(ert-deftest llm-setup-test-nix-model-registry-inventory ()
  "Preserve the authored provider and route inventory."
  (let* ((registry (llm-setup-test--nix-model-registry))
         (providers (alist-get 'providers registry))
         (models (alist-get 'models registry)))
    (dolist (provider providers)
      (should
       (seq-some
        (lambda (model)
          (equal (alist-get 'provider model)
                 (alist-get 'id provider)))
        models)))
    (should
     (llm-setup-test--nix-model
      registry "litellm" "openrouter/moonshotai/kimi-k3"))
    (should
     (llm-setup-test--nix-model
      registry "litellm" "openrouter/qwen/qwen3.7-max"))
    (dolist (id '("gpt-5.6-luna" "gpt-5.6-sol" "gpt-5.6-terra"))
      (should
       (llm-setup-test--nix-model registry "positron-openai" id))
      (should
       (llm-setup-test--nix-model
        registry "litellm" (concat "positron_openai/" id))))))

(ert-deftest llm-setup-test-nix-model-registry-claude-selections ()
  "Publish exact Claude routes and preserve the Haiku-class selection."
  (let* ((registry (llm-setup-test--nix-model-registry))
         (selections (alist-get 'selections registry)))
    (dolist
        (case '((claudeDefault "claude-opus-4-8[1m]")
                (claudeHaiku "claude-sonnet-5")
                (claudeSubagent "claude-opus-4-8")))
      (pcase-let ((`(,role ,model-id) case))
        (let ((selection (alist-get role selections)))
          (should
           (equal (alist-get 'provider selection)
                  "positron-anthropic"))
          (should (equal (alist-get 'model selection) model-id))
          (should
           (llm-setup-test--nix-model
            registry "positron-anthropic" model-id)))))))

(ert-deftest llm-setup-test-nix-model-registry-gpt-5.6-sol-limits ()
  "Preserve GPT-5.6 Sol's long context and output limits in the registry."
  (let ((model
         (llm-setup-test--nix-model
          (llm-setup-test--nix-model-registry)
          "litellm" "positron_openai/gpt-5.6-sol")))
    (should (= 1050000 (alist-get 'contextLimit model)))
    (should (= 128000 (alist-get 'maxOutputTokens model)))
    (should (= 128000 (alist-get 'outputLimit model)))))

(ert-deftest llm-setup-test-nix-model-registry-references-resolve ()
  "Keep provider IDs, routes, and every selected model unambiguous."
  (let* ((registry (llm-setup-test--nix-model-registry))
         (providers (alist-get 'providers registry))
         (models (alist-get 'models registry))
         (provider-ids (mapcar (lambda (p) (alist-get 'id p)) providers))
         (routes
          (mapcar
           (lambda (model)
             (cons (alist-get 'provider model) (alist-get 'id model)))
           models))
         (selection-routes
          (mapcar
           (lambda (selection)
             (cons (alist-get 'provider (cdr selection))
                   (alist-get 'model (cdr selection))))
           (alist-get 'selections registry))))
    (should
     (= (length provider-ids)
        (length (delete-dups (copy-sequence provider-ids)))))
    (should
     (= (length routes)
        (length (delete-dups (copy-tree routes)))))
    (dolist (model models)
      (should (member (alist-get 'provider model) provider-ids)))
    (should
     (equal
      selection-routes
      '(("litellm" . "hera/omlx/Qwen3.6-27B-oQ4e-mtp")
        ("positron-anthropic" . "claude-opus-4-8[1m]")
        ("positron-anthropic" . "claude-sonnet-5")
        ("positron-anthropic" . "claude-opus-4-8"))))
    (dolist (selection-route selection-routes)
      (should (member selection-route routes)))))

(ert-deftest llm-setup-test-nix-model-registry-selection-variables ()
  "Project every model-selection variable into its exact role."
  (cl-progv '(llm-setup-default-provider) '("override-default-provider")
    (should
     (equal
      (alist-get
       'provider
       (alist-get
        'default
        (alist-get
         'selections (llm-setup-test--nix-model-registry))))
      "override-default-provider")))
  (let ((llm-setup-default-instance-name 'override-default-model))
    (should
     (equal
      (alist-get
       'model
       (alist-get
        'default
        (alist-get
         'selections (llm-setup-test--nix-model-registry))))
      "override-default-model")))
  (let ((llm-setup-claude-provider "override-claude-provider"))
    (dolist (role '(claudeDefault claudeHaiku claudeSubagent))
      (should
       (equal
        (alist-get
         'provider
         (alist-get
          role
          (alist-get
           'selections (llm-setup-test--nix-model-registry))))
        "override-claude-provider"))))
  (dolist
      (case '((llm-setup-claude-default-model-id claudeDefault
                                                 "override-claude-default")
              (llm-setup-claude-haiku-model-id claudeHaiku
                                               "override-claude-haiku")
              (llm-setup-claude-subagent-model-id claudeSubagent
                                                  "override-claude-subagent")))
    (pcase-let ((`(,variable ,role ,value) case))
      (cl-progv (list variable) (list value)
        (should
         (equal
          (alist-get
           'model
           (alist-get
            role
            (alist-get
             'selections (llm-setup-test--nix-model-registry))))
          value))))))

(ert-deftest llm-setup-test-nix-model-registry-preserves-source-order ()
  "Keep provider model groups contiguous and in source order."
  (let* ((registry (llm-setup-test--nix-model-registry))
         (providers (alist-get 'providers registry))
         (models (alist-get 'models registry))
         (provider-ids
          (mapcar (lambda (model) (alist-get 'provider model)) models))
         (expected-provider-order
          '("positron-anthropic" "positron-google" "positron-openai"
            "nvidia" "litellm" "llama-cpp-remote" "omlx"
            "llama-cpp-local")))
    (should
     (equal
      (mapcar (lambda (provider) (alist-get 'id provider)) providers)
      expected-provider-order))
    (should
     (equal
      (delete-dups (copy-sequence provider-ids))
      expected-provider-order))
    (dolist (provider-id expected-provider-order)
      (let ((positions
             (cl-loop for model-provider in provider-ids
                      for index from 0
                      when (equal model-provider provider-id)
                      collect index)))
        (should positions)
        (should
         (equal positions
                (number-sequence (car positions) (car (last positions)))))))
    (should
     (equal (llm-setup-render-nix-model-registry)
            (llm-setup-render-nix-model-registry)))))

(ert-deftest llm-setup-test-nix-model-registry-deduplicates-in-source-order ()
  "Deduplicate keys, merge hosts, and use the last declaration's limits."
  (let*
      ((llm-setup-models-list
        (list
         (make-llm-setup-model
          :name 'first
          :context-length 100
          :max-output-tokens 10
          :instances
          (list (make-llm-setup-instance :provider 'local)))
         (make-llm-setup-model
          :name 'shared-family
          :context-length 200
          :max-output-tokens 20
          :instances
          (list
           (make-llm-setup-instance
            :name 'shared
            :provider 'local
            :hostnames '("hera")
            :context-length 201
            :max-output-tokens 21)
           (make-llm-setup-instance
            :name 'shared
            :provider 'local
            :hostnames '("clio")
            :context-length 202
            :max-output-tokens 22)))
         (make-llm-setup-model
          :name 'last
          :context-length 300
          :max-output-tokens 30
          :instances
          (list (make-llm-setup-instance :provider 'local)))))
       (llm-setup-nix-provider-defs
        (list
         (list
          :id "litellm"
          :display-name "LiteLLM"
          :base-url "https://example.invalid/v1/"
          :api-key '((env . "LITELLM_API_KEY"))
          :match-providers '(local)
          :include-limits t
          :default-output-limit 99
          :include-host-filter t)))
       (llm-setup-default-instance-name 'shared)
       (registry (llm-setup-test--nix-model-registry))
       (models (alist-get 'models registry))
       (shared
        (llm-setup-test--nix-model registry "litellm" "shared")))
    (should
     (equal
      (mapcar (lambda (model) (alist-get 'id model)) models)
      '("first" "shared" "last")))
    (should (= 22 (alist-get 'maxOutputTokens shared)))
    (should (= 202 (alist-get 'contextLimit shared)))
    (should (= 99 (alist-get 'outputLimit shared)))
    (should-not (assq 'hosts shared))))

(ert-deftest llm-setup-test-nix-model-registry-projects-hosts ()
  "Project provider and model host restrictions, omitting unrestricted ones."
  (let* ((registry (llm-setup-test--nix-model-registry))
         (remote
          (llm-setup-test--nix-provider registry "llama-cpp-remote"))
         (shared-omlx
          (llm-setup-test--nix-model
           registry "litellm"
           "hera/omlx/cohere-transcribe-03-2026-mlx-fp16"))
         (omlx
          (llm-setup-test--nix-model
           registry "omlx" "cohere-transcribe-03-2026-mlx-fp16"))
         (local-only
          (llm-setup-test--nix-model
           registry "llama-cpp-local" "cohere-transcribe-03-2026"))
         (local-unrestricted
          (llm-setup-test--nix-model
           registry "llama-cpp-local" "Bonsai-8B")))
    (should (equal (alist-get 'hosts remote) '("clio")))
    (dolist (provider (alist-get 'providers registry))
      (unless (equal (alist-get 'id provider) "llama-cpp-remote")
        (should-not (assq 'hosts provider))))
    (should (equal (alist-get 'hosts shared-omlx) '("hera")))
    (should (equal (alist-get 'hosts omlx) '("hera")))
    (should (equal (alist-get 'hosts local-only) '("hera")))
    (should-not (assq 'hosts local-unrestricted))))

(ert-deftest llm-setup-test-nix-model-registry-excludes-embedding-and-reranker ()
  "Exclude embedding and reranker instances while retaining speech routes."
  (let* ((registry (llm-setup-test--nix-model-registry))
         (models (alist-get 'models registry))
         (forbidden
          '("bge-m3" "bge-reranker-v2-m3" "nomic-embed-text-v2-moe"
            "Qwen.Qwen3-Reranker-8B" "Qwen3-Embedding-8B")))
    (dolist (model models)
      (dolist (fragment forbidden)
        (should-not
         (string-match-p
          (regexp-quote fragment) (alist-get 'id model)))))
    (should
     (llm-setup-test--nix-model
      registry "omlx" "cohere-transcribe-03-2026-mlx-fp16"))
    (should
     (llm-setup-test--nix-model
      registry "llama-cpp-local" "granite-speech-4.1-2b"))))

(ert-deftest llm-setup-test-nix-model-registry-deterministic-and-secret-free ()
  "Render deterministically without secrets or file writes."
  (let ((llm-setup-api-key "TASK1-SENTINEL-SECRET"))
    (cl-letf (((symbol-function 'write-region)
               (lambda (&rest _) (ert-fail "write-region must not run")))
              ((symbol-function 'write-file)
               (lambda (&rest _) (ert-fail "write-file must not run")))
              ((symbol-function 'rename-file)
               (lambda (&rest _) (ert-fail "rename-file must not run"))))
      (let ((first-value (llm-setup-nix-model-registry))
            (first-json (llm-setup-render-nix-model-registry)))
        (should (equal first-value (llm-setup-nix-model-registry)))
        (should (equal first-json (llm-setup-render-nix-model-registry)))
        (should (string-suffix-p "\n" first-json))
        (should-not
         (string-match-p (regexp-quote llm-setup-api-key) first-json))))))

(ert-deftest llm-setup-test-build-nix-model-registry-idempotent ()
  "Write deterministic bytes once and leave an identical file untouched."
  (should
   (equal
    llm-setup-nix-model-registry-path
    "~/src/nix/config/ai/model-registry.json"))
  (let* ((directory (make-temp-file "llm-setup-registry-" t))
         (path (expand-file-name "model-registry.json" directory))
         (unused-default (expand-file-name "unused.json" directory))
         (llm-setup-nix-model-registry-path unused-default)
         (first-render (llm-setup-render-nix-model-registry))
         (second-render (llm-setup-render-nix-model-registry))
         (expected
          (with-temp-buffer
            (insert first-render)
            (json-pretty-print-buffer)
            (buffer-string))))
    (unwind-protect
        (progn
          (should
           (equal
            llm-setup-nix-model-registry-path
            unused-default))
          (should (equal first-render second-render))
          (llm-setup-build-nix-model-registry path)
          (should (file-exists-p path))
          (should-not (file-exists-p unused-default))
          (set-file-times path (seconds-to-time 1700000000))
          (let* ((before (file-attributes path 'string))
                 (before-time (file-attribute-modification-time before))
                 (before-inode (file-attribute-inode-number before)))
            (cl-letf (((symbol-function 'make-temp-file)
                       (lambda (&rest _)
                         (ert-fail "An identical write must not create a temp file"))))
              (llm-setup-build-nix-model-registry path))
            (let ((after (file-attributes path 'string)))
              (should
               (equal before-time
                      (file-attribute-modification-time after)))
              (should
               (equal before-inode
                      (file-attribute-inode-number after)))))
          (with-temp-buffer
            (set-buffer-multibyte nil)
            (insert-file-contents-literally path)
            (should
             (equal
              (buffer-string)
              (encode-coding-string expected 'utf-8-unix)))))
      (delete-directory directory t))))

(ert-deftest llm-setup-test-build-nix-model-registry-atomic-replacement ()
  "Replace changed content by same-directory rename using UTF-8 Unix bytes."
  (let* ((directory (make-temp-file "llm-setup-registry-" t))
         (path (expand-file-name "model-registry.json" directory))
         (expected
          (encode-coding-string
           "{\n  \"name\": \"café\"\n}\n" 'utf-8-unix))
         (original-pretty (symbol-function 'json-pretty-print-buffer))
         (original-rename (symbol-function 'rename-file))
         (pretty-calls 0)
         renamed-temp
         old-inode)
    (unwind-protect
        (progn
          (write-region "old\n" nil path nil 'silent)
          (setq old-inode
                (file-attribute-inode-number (file-attributes path 'string)))
          (cl-letf (((symbol-function 'llm-setup-render-nix-model-registry)
                     (lambda () "{\"name\":\"café\"}\n"))
                    ((symbol-function 'json-pretty-print-buffer)
                     (lambda ()
                       (cl-incf pretty-calls)
                       (should
                        (equal (buffer-string) "{\"name\":\"café\"}\n"))
                       (funcall original-pretty)))
                    ((symbol-function 'rename-file)
                     (lambda (source destination &optional replace)
                       (setq renamed-temp source)
                       (should
                        (equal (file-name-directory source)
                               (file-name-directory destination)))
                       (should (equal destination path))
                       (should replace)
                       (with-temp-buffer
                         (set-buffer-multibyte nil)
                         (insert-file-contents-literally source)
                         (should (equal (buffer-string) expected)))
                       (with-temp-buffer
                         (insert-file-contents path)
                         (should (equal (buffer-string) "old\n")))
                       (funcall original-rename source destination replace))))
            (llm-setup-build-nix-model-registry path))
          (should (= 1 pretty-calls))
          (should renamed-temp)
          (should-not (file-exists-p renamed-temp))
          (should-not
           (equal
            old-inode
            (file-attribute-inode-number (file-attributes path 'string))))
          (with-temp-buffer
            (set-buffer-multibyte nil)
            (insert-file-contents-literally path)
            (should (equal (buffer-string) expected))
            (should-not (string-match-p "\r" (buffer-string)))))
      (delete-directory directory t))))

(ert-deftest llm-setup-test-build-nix-model-registry-cleans-temp-on-error ()
  "Delete the same-directory temporary file when atomic replacement fails."
  (let* ((directory (make-temp-file "llm-setup-registry-" t))
         (path (expand-file-name "model-registry.json" directory))
         captured-temp)
    (unwind-protect
        (progn
          (write-region "old\n" nil path nil 'silent)
          (cl-letf (((symbol-function 'llm-setup-render-nix-model-registry)
                     (lambda () "{\"new\":true}\n"))
                    ((symbol-function 'rename-file)
                     (lambda (source _destination &optional _replace)
                       (setq captured-temp source)
                       (should (file-exists-p source))
                       (error "Simulated rename failure"))))
            (should-error
             (llm-setup-build-nix-model-registry path)
             :type 'error))
          (should captured-temp)
          (should-not (file-exists-p captured-temp))
          (with-temp-buffer
            (insert-file-contents path)
            (should (equal (buffer-string) "old\n")))
          (should
           (equal
            (directory-files directory nil directory-files-no-dot-files-regexp)
            '("model-registry.json"))))
      (delete-directory directory t))))

(ert-deftest llm-setup-test-reset-orchestration ()
  "Run hera/clio llama-swap, Nix registry, and GPTel updates in order."
  (let (events)
    (cl-progv '(gptel-model gptel-backend) '(nil nil)
      (cl-letf (((symbol-function 'llm-setup-check-instances)
                 (lambda () (push 'check events) 0))
                ((symbol-function 'llm-setup-build-llama-swap-yaml)
                 (lambda (&optional hostname)
                   (push (list 'llama-swap hostname) events)))
                ((symbol-function 'llm-setup-build-nix-model-registry)
                 (lambda (&optional _path) (push 'nix-registry events)))
                ((symbol-function 'gptel-backends-make-litellm)
                 (lambda () (push 'gptel events) 'test-backend)))
        (llm-setup-reset)
        (should
         (equal
          (nreverse events)
          '(check
            (llama-swap nil)
            (llama-swap "clio")
            nix-registry
            gptel)))
        (should
         (eq (symbol-value 'gptel-model) llm-setup-default-instance-name))
        (should (eq (symbol-value 'gptel-backend) 'test-backend))))))

(provide 'llm-setup-test)

;;; llm-setup-test.el ends here
