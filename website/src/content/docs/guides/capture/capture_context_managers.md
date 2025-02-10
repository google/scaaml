---
title: Capture Context Managers
description: Capture Context Managers
---

After a crash of a capture campaign we might need to release resources (e.g., call `scope.dis()`).
Python context-managers (see a [tutorial on context-managers](https://book.pythontips.com/en/latest/context_managers.html)) help us to ensure resources are always released even in the case of capture in a script running on remote computer.

# Automation Scripts

When using a JupyterLab notebook we can allways ensure `scope.dis()` is called.
However when a Python script crashes we no longer hold the `scope` variable and resuming capture is much more complicated.
We could wrap everything by `try-finally` blocks.
One of the advantages of using context managers is they prevent us from forgetting to add the `finally` block.

## Try-Finally Solution

```python
import time

import chipwhisperer as cw


def main():
    try:
        scope = cw.scope()
        scope.default_setup()
        time.sleep(1)
        target = cw.target(scope)

        target.con()
        time.sleep(1)
        scope.io.target_pwr = False
        time.sleep(1)
        scope.io.target_pwr = True

        ktp = cw.ktp.Basic()
        key, pt = ktp.new_pair()
        trace = cw.capture_trace(scope, target, pt, key)
        print(trace)
    finally:
        target.dis()
        scope.dis()


if __name__ == "__main__":
    main()
```

## Context-Managers Solution

TODO
