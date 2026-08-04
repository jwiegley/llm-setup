;;; llm-setup-test.el --- Tests for llm-setup  -*- lexical-binding: t; -*-

;;; Commentary:

;; Regression tests for generated model configuration.

;;; Code:

(require 'ert)
(require 'llm-setup)

(declare-function llm-setup-aider-model-name "llm-setup")
(declare-function llm-setup--instance-eligible-for-host-p "llm-setup")

(ert-deftest llm-setup-test-sync-mlx-cache-disabled ()
  "Ignore cached and registered MLX models when cache sync is disabled."
  (let* ((llm-setup-sync-mlx-cache nil)
         (model (make-llm-setup-model :name 'cached-mlx))
         (instance
          (make-llm-setup-instance
           :name 'organization/cached-mlx-4bit
           :engine 'vllm-mlx))
         (instances (list (cons model instance))))
    (cl-letf (((symbol-function 'file-directory-p)
               (lambda (&rest _)
                 (ert-fail "Disabled MLX sync must not inspect the cache"))))
      (let ((discovered (llm-setup-sync--discover-mlx))
            (known (llm-setup-sync--known-mlx-names instances)))
        (should (zerop (hash-table-count discovered)))
        (should (zerop (hash-table-count known)))
        (should
         (equal (llm-setup-sync--compare-mlx discovered known)
                '(:new nil :dead nil)))))))

(ert-deftest llm-setup-test-model-list-sorted-and-unique ()
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
                  (Qwen3.8-Max
                   qwen/qwen3.8-max
                   openrouter/qwen/qwen3.8-max
                   1048576)))
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
      (should (= 262144 (llm-setup-model-context-length model)))
      (should (= 1.0 (llm-setup-model-temperature model)))
      (should
       (equal (llm-setup-instance-model-path local)
              "~/Models/unsloth_GLM-5.2-GGUF"))
      (should (= 262144
                 (llm-setup-get-instance-context-length model local)))
      (should (eq (llm-setup-instance-name openrouter) 'z-ai/glm-5.2))
      (should (= 1048576 (llm-setup-instance-context-length openrouter)))
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
  "Require the default to resolve through hera's oMLX backend."
  (should
   (eq llm-setup-default-instance-name
       'DeepSeek-V4-Flash-0731-oQ8e-mtp))
  (should (equal (llm-setup-aider-model-name)
                 "openai/DeepSeek-V4-Flash-0731-oQ8e-mtp"))
  (should
   (= 1
      (cl-count
       llm-setup-default-instance-name
       (mapcar #'car (llm-setup-gptel-backends "hera"))
       :test #'eq))))

(ert-deftest llm-setup-test-reset-orchestration ()
  "Run validation, hera/clio llama-swap, and GPTel updates in order."
  (let (events)
    (cl-progv '(gptel-model gptel-backend) '(nil nil)
      (cl-letf (((symbol-function 'llm-setup-check-instances)
                 (lambda () (push 'check events) 0))
                ((symbol-function 'llm-setup-build-llama-swap-yaml)
                 (lambda (&optional hostname)
                   (push (list 'llama-swap hostname) events)))
                ((symbol-function 'gptel-backends-omlx)
                 (lambda () (push 'gptel events) 'test-backend)))
        (llm-setup-reset)
        (should
         (equal
          (nreverse events)
          '(check
            (llama-swap nil)
            (llama-swap "clio")
            gptel)))
        (should
         (eq (symbol-value 'gptel-model) llm-setup-default-instance-name))
        (should (eq (symbol-value 'gptel-backend) 'test-backend))))))

(provide 'llm-setup-test)

;;; llm-setup-test.el ends here
