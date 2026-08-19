/**
 * Shared shell for the routes that are not built yet.
 *
 * OWNED-BY: whichever agent builds that screen. Delete this component once no
 * route uses it.
 */
interface PlaceholderProps {
  title: string;
  /** One sentence on what this screen will do, in the app's own voice. */
  description: string;
  /** Which milestone in docs/DESIGN.md builds it. */
  milestone: string;
}

export default function Placeholder({
  title,
  description,
  milestone,
}: PlaceholderProps) {
  return (
    <section className="mx-auto flex w-full max-w-2xl flex-col gap-3 px-8 py-14">
      <h1 className="text-2xl font-semibold tracking-tight">{title}</h1>
      <p className="text-ink-soft">{description}</p>
      <p className="echo-chip w-fit">Not built yet &middot; {milestone}</p>
    </section>
  );
}
