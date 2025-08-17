import streamlit as st
import imaplib
import email
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# ---------------------------
# Load pre-trained spam model
# ---------------------------
model = pickle.load(open("spam_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("📧 Gmail Spam Classifier")
st.write("Enter your Gmail credentials to classify unread emails as Spam or Not Spam.")

email_user = st.text_input("Gmail Address")
email_pass = st.text_input("App Password", type="password")
num_emails = st.slider("Number of emails to fetch", min_value=10, max_value=200, value=50)

if st.button("Fetch & Classify Emails"):
    if not email_user or not email_pass:
        st.warning("Please enter both Gmail and App Password.")
    else:
        try:
            # Connect to Gmail IMAP
            mail = imaplib.IMAP4_SSL("imap.gmail.com")
            mail.login(email_user, email_pass)
            mail.select("inbox")

            # Fetch only unread emails
            status, messages = mail.search(None, "UNSEEN")
            email_ids = messages[0].split()

            results = []

            st.info(f"Found {len(email_ids)} unread emails. Classifying...")

            # Limit number of emails
            for e_id in email_ids[:num_emails]:
                status, msg_data = mail.fetch(e_id, "(RFC822)")
                raw_email = msg_data[0][1]
                msg = email.message_from_bytes(raw_email)

                subject = msg["subject"] or ""
                body = ""
                if msg.is_multipart():
                    for part in msg.walk():
                        if part.get_content_type() == "text/plain":
                            try:
                                body += part.get_payload(decode=True).decode()
                            except:
                                pass
                else:
                    try:
                        body = msg.get_payload(decode=True).decode()
                    except:
                        pass

                full_text = f"{subject} {body}"
                input_vec = vectorizer.transform([full_text])
                prediction = model.predict(input_vec)[0]

                results.append({
                    "From": msg["from"],
                    "Subject": subject,
                    "Body": body,
                    "Spam Status": "🚫 SPAM" if prediction == 1 else "✅ NOT SPAM"
                })

            mail.logout()

            # Convert to DataFrame
            df = pd.DataFrame(results)

            # Highlight spam in red, non-spam in green
            def highlight_spam(row):
                color = 'background-color: red' if row['Spam Status'] == '🚫 SPAM' else 'background-color: lightgreen'
                return [color]*len(row)

            st.dataframe(df[['From','Subject','Spam Status']].style.apply(highlight_spam, axis=1))

            # Show email body on click
            for idx, row in df.iterrows():
                with st.expander(f"Email from: {row['From']} | Subject: {row['Subject']} | Status: {row['Spam Status']}"):
                    st.write(row['Body'])

            # Allow CSV download
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download CSV", data=csv, file_name="classified_emails.csv", mime="text/csv")

        except Exception as e:
            st.error(f"Error: {e}")