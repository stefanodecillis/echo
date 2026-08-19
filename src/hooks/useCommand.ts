import { useCallback, useRef, useState } from "react";

import { toUiError } from "../lib/ipc";
import type { UiError } from "../lib/types";

export type CommandStatus = "idle" | "loading" | "success" | "error";

export interface UseCommandResult<Args extends unknown[], T> {
  data: T | undefined;
  error: UiError | undefined;
  status: CommandStatus;
  loading: boolean;
  /** Call with the same arguments the wrapped function takes. Rethrows the
   * normalised `UiError` so a caller can also react locally (e.g. keep a form
   * open), but the hook's own `error` state is always set first. */
  run: (...args: Args) => Promise<T>;
  reset: () => void;
}

/**
 * Wrap one `lib/ipc` call with loading/error/data state.
 *
 * ```ts
 * const { run, loading, error } = useCommand(startRecording);
 * <Button loading={loading} onClick={() => run()}>Start</Button>
 * {error && <p>{error.message}</p>}
 * ```
 */
export function useCommand<Args extends unknown[], T>(
  fn: (...args: Args) => Promise<T>,
): UseCommandResult<Args, T> {
  const [status, setStatus] = useState<CommandStatus>("idle");
  const [data, setData] = useState<T>();
  const [error, setError] = useState<UiError>();
  const fnRef = useRef(fn);
  fnRef.current = fn;

  const run = useCallback(async (...args: Args) => {
    setStatus("loading");
    setError(undefined);
    try {
      const result = await fnRef.current(...args);
      setData(result);
      setStatus("success");
      return result;
    } catch (err) {
      const uiError = toUiError(err);
      setError(uiError);
      setStatus("error");
      throw uiError;
    }
  }, []);

  const reset = useCallback(() => {
    setStatus("idle");
    setData(undefined);
    setError(undefined);
  }, []);

  return { data, error, status, loading: status === "loading", run, reset };
}
