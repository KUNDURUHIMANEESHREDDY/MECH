/**
 * Peer Review System.
 */

export class ReviewSystem {
  constructor() {
    this.reviews = [
      { id: 'rev_1', artifactId: 'disc_1', author: 'Dr. Carol', status: 'Approved', comment: 'Methodology and causal p-values validated cleanly.' }
    ];
  }

  submitReview({ artifactId, author, status, comment }) {
    const rev = {
      id: `rev_${this.reviews.length + 1}`,
      artifactId,
      author,
      status, // Approved, Rejected, NeedsRevision
      comment,
      submittedAt: new Date().toISOString()
    };
    this.reviews.push(rev);
    return rev;
  }

  listReviews(artifactId) {
    if (!artifactId) return [...this.reviews];
    return this.reviews.filter((r) => r.artifactId === artifactId);
  }
}
