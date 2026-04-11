import { useEffect, useRef } from "react";
/**
 * 조건부 폴링 훅.
 * active가 true일 때만 intervalMs 간격으로 fn을 호출.
 */
export function usePolling(fn, intervalMs, active) {
    const savedFn = useRef(fn);
    savedFn.current = fn;
    useEffect(() => {
        if (!active)
            return;
        savedFn.current();
        const id = setInterval(() => savedFn.current(), intervalMs);
        return () => clearInterval(id);
    }, [intervalMs, active]);
}
