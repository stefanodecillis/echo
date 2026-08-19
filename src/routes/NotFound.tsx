import { Link } from "react-router-dom";

import { common } from "@/lib/copy";

/** Any URL that does not exist. Should be unreachable in a desktop app. */
export default function NotFound() {
  return (
    <section className="mx-auto flex w-full max-w-2xl flex-col gap-3 px-8 py-14">
      <h1 className="text-2xl font-semibold tracking-tight">{common.notFoundTitle}</h1>
      <p className="text-ink-soft">{common.notFoundBody}</p>
      <Link to="/" className="echo-pill-primary w-fit">
        {common.backToHome}
      </Link>
    </section>
  );
}
