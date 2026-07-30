/**
 * User Account & Role Permission Service.
 */

export class UserAccountService {
  constructor() {
    this.users = [
      { id: 'usr_1', name: 'Dr. Alice', email: 'alice@lab.ai', role: 'Owner' },
      { id: 'usr_2', name: 'Dr. Bob', email: 'bob@lab.ai', role: 'Researcher' },
      { id: 'usr_3', name: 'Dr. Carol', email: 'carol@lab.ai', role: 'Reviewer' }
    ];
  }

  listUsers() {
    return [...this.users];
  }

  hasPermission(role, action) {
    if (role === 'Owner' || role === 'Admin') return true;
    if (role === 'Researcher' && action !== 'delete_workspace') return true;
    if (role === 'Reviewer' && (action === 'view' || action === 'comment' || action === 'approve')) return true;
    return false;
  }
}
