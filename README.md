# PR-Reviewer 🕵🏼

![PR Reviewer](images/PR_review_process.png)

Streamline Your GitHub Pull Requests with AI, co-authored with LLM(Large language model).

🚀 Excited to announce our new open-source project: PR Review Bot, a GitHub Pull Request review bot powered by LLM.

🤖 PR Review Bot automatically reviews open PRs in your GitHub repository, providing helpful feedback and even approving or requesting changes based on the analysis of the PR text and comments.

🔧 Save time and effort in your development workflow by automating the initial review process, ensuring PRs adhere to your project's guidelines and best practices.

🌟 Key features:
- Automatically reviews open PRs
- Leverages LLM for intelligent analysis and feedback
- Can be easily customized to fit your project needs
- Easy to set up and use
- Cost Analysis of each review

Code review flow
1. **Creating**: Authors start modifying, adding, or deleting some code; once ready, they create a change.
2. **Previewing**: Authors then use Critique to view the diff of the change and view the results of automatic code analyzers  When they are ready, authors mail the change out to one or more reviewers.
3. **Commenting**: Reviewers view the diff in the web UI, drafting comments as they go. Program analysis results, if present, are also visible to reviewers. Unresolved comments represent action items for the change author to definitely address. Resolved comments include optional or informational comments that may not require any action by a change author.
4. **Addressing Feedback**: The author now addresses comments, either by updating the change or by replying to comments. When a change has been updated, the author uploads a new snapshot. The author and reviewers can look at diffs between any pairs of snapshots to see what changed.
5. **Approving**: Once all comments have been addressed, the reviewers approve the change and mark it ‘**LGTM**’ (Looks Good To Me). To eventually commit a change, a developer typically must have approval from at least one reviewer. Usually, only one reviewer is required to satisfy the aforementioned requirements of ownership and readability.
