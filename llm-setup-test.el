;;; llm-setup-test.el --- Tests for llm-setup  -*- lexical-binding: t; -*-

;;; Commentary:

;; Regression tests for generated model configuration.

;;; Code:

(require 'ert)
(require 'llm-setup)

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
