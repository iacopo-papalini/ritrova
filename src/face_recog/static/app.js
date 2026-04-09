async function assignFace(event, faceId) {
  event.preventDefault();
  const form = event.target;
  const select = form.querySelector('select');
  const personId = select.value;
  if (!personId) return false;

  const body = new FormData();
  body.append('person_id', personId);

  const resp = await fetch(`/api/faces/${faceId}/assign`, {method: 'POST', body});
  if (resp.ok) {
    location.reload();
  }
  return false;
}

async function unassignFace(faceId) {
  const resp = await fetch(`/api/faces/${faceId}/unassign`, {method: 'POST'});
  if (resp.ok) {
    location.reload();
  }
}
