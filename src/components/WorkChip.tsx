import { jobLine, labels, workChip } from "../lib/copy";
import { presentJob } from "../lib/jobs";
import type { MeetingWork } from "../lib/meetingWork";
import { Chip } from "./Chip";
import { WorkingRing } from "./WorkingRing";

export interface WorkChipProps {
  work: MeetingWork;
}

/**
 * What one meeting row says about work Echo has not finished on it.
 *
 * ONE COMPONENT, FIVE STATES
 * For the same reason `jobSentence` is one function: this is rendered from two
 * different lists, and three ternaries in two places drift apart. Every state a
 * row can be in is decided in `lib/meetingWork.ts` and drawn here, and nowhere
 * else picks between them.
 *
 * HOW RUNNING AND WAITING ARE TOLD APART
 * Not by colour. There is no green or amber in this app and `live` red means
 * recording or error, so three things that are not hue carry it instead: the
 * running chip moves, it has a border where the others are flat, and its text is
 * a step darker than the row around it. The waiting chip is exactly the weight of
 * the language chip beside it, which is the point — it is another quiet fact
 * about the meeting, not an event.
 *
 * NO PROGRESS BAR, IN ANY STATE
 * For waiting, because of the rule at the top of `lib/jobs.ts`: a job that has
 * not started has made no progress, and a bar would say otherwise. For running,
 * because half a dozen moving bars in a list is the noise docs/DESIGN.md §4 is
 * written against, and because the map these rows read deliberately does not
 * carry a fraction. The bar belongs on the meeting's own screen. Please don't add
 * one here.
 *
 * NO `aria-live`
 * Half a dozen live regions in a list would announce every job transition in the
 * app to somebody reading something else entirely. The chip is plain text inside
 * a link that is already labelled by the meeting's title.
 */
export function WorkChip({ work }: WorkChipProps) {
  switch (work.state) {
    case "none":
      return null;

    case "running":
      // Names what is happening, because which of the four passes is running is
      // the whole thing somebody wants from a glance.
      return (
        <Chip variant="outline" className="min-w-0 text-ink-soft" icon={<WorkingRing size="sm" />}>
          <span className="min-w-0 truncate">{jobLine.running(presentJob(work.job).label)}</span>
        </Chip>
      );

    case "waiting":
      return (
        <Chip className="min-w-0">
          <span className="min-w-0 truncate">{workChip.waiting}</span>
        </Chip>
      );

    case "deferred":
      return (
        <Chip className="min-w-0">
          <span className="min-w-0 truncate">{workChip.deferred}</span>
        </Chip>
      );

    case "stopped":
      // Names the pass that stopped: it is what tells a person which part of this
      // meeting is missing, which is the only actionable thing here. `live` is
      // recording *or* error, and this is an error.
      return (
        <Chip variant="danger" className="min-w-0">
          <span className="min-w-0 truncate">{jobLine.stopped(presentJob(work.job).label)}</span>
        </Chip>
      );

    case "unknown":
      // Says something true about the meeting and claims no activity at all.
      return (
        <Chip className="min-w-0">
          <span className="min-w-0 truncate">{labels.meetingStatus.processing}</span>
        </Chip>
      );
  }
}
